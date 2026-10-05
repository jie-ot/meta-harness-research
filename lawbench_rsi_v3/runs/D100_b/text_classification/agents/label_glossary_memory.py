"""Label Glossary Memory — anchors predictions to canonical label strings.

Diagnosis from score diagnostics (71 wrong out of 100):
  - Surface-form near-misses: model predicts a paraphrase of the correct label
    instead of the exact canonical string (e.g. "销售假冒注册商标商品" vs
    "销售假冒注册商标的商品", "非法种植罂粟" vs "非法种植毒品原植物").
  - Specificity errors: model predicts a general label when a specific one is
    correct ("诈骗" vs "合同诈骗", "行贿" vs "单位行贿").

Root cause: the base system's prompt shows examples in Q/A format but gives
the LLM no inventory of exact canonical label strings. The LLM paraphrases
from memory rather than copying the precise string it has seen in training.

This system accumulates every ground-truth label string encountered during
learn_from_batch and injects them as a reference glossary section in the
prompt, after the examples. The LLM is explicitly instructed to choose from
that list. At cold start (no labels learned yet) the prompt degrades
gracefully to the base format without a glossary section.

Retrieval uses CJK character bigrams for genuine similarity over Chinese text
(same fix as cjk_bigram_confusion_memory). The novelty here is the prompt
architecture: a live-built glossary as a grounding constraint.
"""

import json
import re
from collections import defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# Prompt when we have a glossary to anchor to
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

# Prompt when no glossary is available yet (cold start)
PROMPT_NO_GLOSSARY = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000
# Reserve chars for glossary and prompt boilerplate
_EXAMPLE_BUDGET = 24000


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


class LabelGlossaryMemory(MemorySystem):
    """Retrieval with a live canonical-label glossary to prevent near-miss errors."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        # Stored training examples for retrieval
        self.examples: list[dict[str, Any]] = []
        # Ordered list of unique canonical label strings seen during training
        self.label_glossary: list[str] = []

    # ------------------------------------------------------------------
    # Glossary management
    # ------------------------------------------------------------------

    def _add_labels(self, ground_truth: str) -> None:
        """Extract individual labels from a (possibly multi-label) ground truth."""
        for lbl in ground_truth.split(';'):
            lbl = lbl.strip()
            if lbl and lbl not in self.label_glossary:
                self.label_glossary.append(lbl)

    def _build_glossary_str(self) -> str:
        return '\n'.join(f'- {lbl}' for lbl in sorted(self.label_glossary))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []
        q_tok = _tokenize(query)
        ranked = sorted(
            [(_jaccard(q_tok, ex['tokens']), idx, ex)
             for idx, ex in enumerate(self.examples)],
            reverse=True,
            key=lambda x: (x[0], x[1])
        )
        parts: list[str] = []
        total = 0
        for _, _, ex in ranked:
            q = ex.get('raw_question', ex['input'])
            part = f"Q: {q}\nA: {ex['target']}"
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
            # Always accumulate the ground truth label into the glossary
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
