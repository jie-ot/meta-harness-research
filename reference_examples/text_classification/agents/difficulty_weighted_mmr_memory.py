"""Difficulty-Weighted MMR Memory.

A new retrieval mechanism that integrates per-label historical error rate directly
into the MMR scoring objective. No prior system has used a continuous difficulty
signal inside the selection loop itself.

Mechanism:
- During learn_from_batch, track per-label error counts and totals.
  difficulty[label] = errors / total_seen (0 = never wrong, 1 = always wrong).

- At predict time, run a single-pass MMR selection where the relevance component
  of each candidate is boosted by its label's difficulty:

      boosted_relevance = jaccard(query, example) * (1 + beta * difficulty[label])
      mmr_score = lambda * boosted_relevance - (1 - lambda) * max_redundancy

  This means a hard-to-predict label with moderate query similarity can outrank
  an easy label with high similarity, because the boost compensates for weaker
  lexical match. Hard-label examples compete directly in the single unified MMR
  pass — no separate phase, no slot reservation.

Why this differs from every prior system:
- label_champion, mmr, label_balanced all use flat scores with no error-rate signal.
- confusion_disambiguation_memory reserves dedicated disambiguation slots (binary),
  then falls back to flat MMR fill.
- This system applies a continuous, graded boost inside the MMR objective itself:
  every example from every label competes in one pass, with difficulty acting as
  a relevance multiplier rather than a slot-allocation rule.

Cold start: all difficulties default to 0 → pure MMR on Jaccard similarity.
"""

import json
import re
from collections import defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000
_MMR_LAMBDA = 0.6       # weight for relevance vs redundancy
_DIFFICULTY_BETA = 1.5  # boost multiplier: score *= (1 + beta * difficulty)


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class DifficultyWeightedMmrMemory(MemorySystem):
    """MMR retrieval with per-label difficulty boost on the relevance term."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        # per-label error tracking
        self._label_errors: dict[str, int] = defaultdict(int)
        self._label_totals: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _difficulty(self, label: str) -> float:
        total = self._label_totals.get(label, 0)
        if total == 0:
            return 0.0
        return self._label_errors.get(label, 0) / total

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        q_tok = _tokenize(query)

        # Pre-compute boosted relevance for every candidate
        candidates: list[tuple[float, int, dict]] = []
        for idx, ex in enumerate(self.examples):
            jac = _jaccard(q_tok, ex["tokens"])
            diff = self._difficulty(ex["target"])
            boosted = jac * (1.0 + _DIFFICULTY_BETA * diff)
            candidates.append((boosted, idx, ex))

        parts: list[str] = []
        selected_tokens: list[frozenset[str]] = []
        total_chars = 0

        while candidates and total_chars < MAX_CHARS:
            best_score = -999.0
            best_item: tuple[float, int, dict] | None = None

            for boosted_rel, i, ex in candidates:
                q = ex.get("raw_question", ex["input"])
                part = f"Q: {q}\nA: {ex['target']}"
                if total_chars + len(part) + 2 > MAX_CHARS:
                    continue
                redundancy = max(
                    (_jaccard(ex["tokens"], st) for st in selected_tokens),
                    default=0.0,
                )
                score = _MMR_LAMBDA * boosted_rel - (1.0 - _MMR_LAMBDA) * redundancy
                if score > best_score:
                    best_score = score
                    best_item = (boosted_rel, i, ex)

            if best_item is None:
                break
            _, idx, ex = best_item
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            parts.append(part)
            selected_tokens.append(ex["tokens"])
            total_chars += len(part) + 2
            candidates = [(s, i, e) for s, i, e in candidates if i != idx]

        return parts

    # ------------------------------------------------------------------
    # MemorySystem interface
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = self._build_parts(input)
        prompt = PROMPT_TEMPLATE.format(
            examples_section="\n\n".join(parts),
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_selected": len(parts),
            "num_labels_tracked": len(self._label_totals),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            ex: dict[str, Any] = {
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": _tokenize(raw_q),
            }
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

            gt = r["ground_truth"]
            self._label_totals[gt] += 1
            if not r.get("was_correct", True):
                self._label_errors[gt] += 1

    def get_context_length(self) -> int:
        return sum(len(p) + 2 for p in self._build_parts(""))

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        return json.dumps({
            "examples": serialisable,
            "label_errors": dict(self._label_errors),
            "label_totals": dict(self._label_totals),
        }, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        for ex in data.get("examples", []):
            restored = dict(ex)
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                raw_q = restored.get("raw_question", restored.get("input", ""))
                restored["tokens"] = _tokenize(raw_q)
            self.examples.append(restored)

        self._label_errors = defaultdict(int, data.get("label_errors", {}))
        self._label_totals = defaultdict(int, data.get("label_totals", {}))
