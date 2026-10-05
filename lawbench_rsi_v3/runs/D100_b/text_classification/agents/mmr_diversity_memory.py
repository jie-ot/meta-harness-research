"""MMR Diversity Memory — diversity-aware retrieval via Maximal Marginal Relevance (axis C).

Build on label_glossary_memory (frontier, 46/100).

Diagnosis of remaining failures in the frontier:
  - Multi-label omissions (~25 of 54 wrong): the model outputs one charge when the
    fact pattern supports two or three.  Pure Jaccard top-k retrieval clusters around
    the single most-similar neighbourhood, so the prompt is saturated with single-charge
    examples from that cluster and contains no examples showing the secondary charge.
  - Over-prediction (~8 cases): the converse — model adds a spurious charge because
    no diverse counter-example demonstrates that that charge is absent here.

New mechanism (axis C):
  Replace Jaccard top-k with Maximal Marginal Relevance (MMR).  MMR iteratively
  selects examples that are (a) relevant to the query AND (b) dissimilar to already-
  selected examples.  The balance is controlled by lam ∈ [0,1]:

      score(i) = lam * relevance(i, query) - (1-lam) * max_sim(i, already_selected)

  With lam=0.6 the selected set spans multiple charge families, exposing the model
  to both single- and multi-charge patterns in the same prompt instead of 8 near-
  identical single-charge examples from one cluster.  This should reduce multi-label
  omissions by ensuring at least one multi-charge exemplar appears when the query
  facts mention concurrent behaviours.

Self-critique: retrieve logic in _build_parts is completely replaced — Jaccard top-k
  is gone; MMR is an iterative greedy algorithm with a fundamentally different objective.
  learn_from_batch is unchanged from the frontier (axis C only, not F).
  This is axis C (selection algorithm), not tried in any prior iteration.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# ---------------------------------------------------------------------------
# Prompts (identical to label_glossary_memory — novel part is retrieval only)
# ---------------------------------------------------------------------------

PROMPT_WITH_GLOSSARY = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Canonical charge labels seen in training (use exact text from this list):**
{glossary}

**Instructions:**
- Analyse the facts and identify the charge(s)
- Your final_answer MUST use exact label text from the list above
- For multiple charges use semicolons: Label1;Label2
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[exact label(s) from list]"}}"""

PROMPT_NO_GLOSSARY = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

_EXAMPLE_BUDGET = 24000   # chars reserved for examples
_MMR_LAMBDA = 0.6         # relevance/diversity balance; 1.0 = pure Jaccard top-k
_MMR_POOL = 60            # pre-filter to this many by raw Jaccard before MMR sweep


def _tokenize(text: str) -> frozenset:
    """CJK character bigrams plus ASCII tokens."""
    cjk_chars = re.findall(r'[一-鿿㐀-䶿]', text)
    bigrams = frozenset(
        cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)
    )
    ascii_toks = frozenset(re.findall(r'[A-Za-z0-9]+', text.lower()))
    return bigrams | ascii_toks


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _mmr_select(
    query_toks: frozenset,
    candidates: list[dict[str, Any]],
    k: int,
    lam: float,
) -> list[int]:
    """Return up to k indices via MMR greedy selection.

    Pre-compute relevance scores once; at each step pick the candidate maximising
        lam * relevance - (1-lam) * max_similarity_to_selected
    """
    if not candidates:
        return []
    relevance = [_jaccard(query_toks, c['tokens']) for c in candidates]
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(k, len(candidates))):
        best_i, best_score = None, float('-inf')
        for i in remaining:
            if selected:
                max_sim = max(
                    _jaccard(candidates[i]['tokens'], candidates[s]['tokens'])
                    for s in selected
                )
            else:
                max_sim = 0.0
            score = lam * relevance[i] - (1.0 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


class MMRDiversityMemory(MemorySystem):
    """Retrieval with MMR diversity to improve multi-label coverage."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.label_glossary: list[str] = []

    # ------------------------------------------------------------------
    # Glossary management
    # ------------------------------------------------------------------

    def _add_labels(self, ground_truth: str) -> None:
        for lbl in ground_truth.split(';'):
            lbl = lbl.strip()
            if lbl and lbl not in self.label_glossary:
                self.label_glossary.append(lbl)

    def _build_glossary_str(self) -> str:
        return '\n'.join(f'- {lbl}' for lbl in sorted(self.label_glossary))

    # ------------------------------------------------------------------
    # MMR retrieval
    # ------------------------------------------------------------------

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []
        q_tok = _tokenize(query)

        # Pre-filter: keep top-_MMR_POOL by raw Jaccard to bound MMR cost O(pool*k)
        pool_ranked = sorted(
            range(len(self.examples)),
            key=lambda i: _jaccard(q_tok, self.examples[i]['tokens']),
            reverse=True,
        )[:_MMR_POOL]
        pool = [self.examples[i] for i in pool_ranked]

        # Estimate how many examples fit in the budget to use as k
        avg_part_len = 300  # conservative estimate for a typical example
        k_estimate = max(1, _EXAMPLE_BUDGET // avg_part_len)

        selected_indices = _mmr_select(q_tok, pool, k=k_estimate, lam=_MMR_LAMBDA)

        parts: list[str] = []
        total = 0
        for idx in selected_indices:
            ex = pool[idx]
            q_text = ex.get('raw_question', ex['input'])
            part = f"Q: {q_text}\nA: {ex['target']}"
            if total + len(part) + 2 > _EXAMPLE_BUDGET:
                break
            parts.append(part)
            total += len(part) + 2
        return parts

    # ------------------------------------------------------------------
    # MemorySystem interface
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = self._build_parts(input)
        examples_section = '\n\n'.join(parts)

        if self.label_glossary:
            glossary_str = self._build_glossary_str()
            prompt = PROMPT_WITH_GLOSSARY.format(
                examples_section=examples_section,
                input=input,
                glossary=glossary_str,
            )
        else:
            prompt = PROMPT_NO_GLOSSARY.format(
                examples_section=examples_section,
                input=input,
            )

        response = self.call_llm(prompt)
        answer = extract_json_field(response, 'final_answer')
        return answer, {
            'full_response': response,
            'num_examples': len(self.examples),
            'num_selected': len(parts),
            'num_glossary_labels': len(self.label_glossary),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw_q = r.get('raw_question', r['input'])
            ex: dict[str, Any] = {
                'input': r['input'],
                'target': r['ground_truth'],
                'tokens': _tokenize(raw_q),
            }
            if 'raw_question' in r:
                ex['raw_question'] = r['raw_question']
            self.examples.append(ex)
            self._add_labels(r['ground_truth'])

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        return json.dumps({
            'examples': serialisable,
            'label_glossary': self.label_glossary,
        }, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        for ex in data.get('examples', []):
            restored = dict(ex)
            raw_q = restored.get('raw_question', restored.get('input', ''))
            restored['tokens'] = _tokenize(raw_q)
            self.examples.append(restored)
        self.label_glossary = data.get('label_glossary', [])
