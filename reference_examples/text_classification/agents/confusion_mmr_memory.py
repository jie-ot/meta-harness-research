"""Confusion-Disambiguation + MMR Fill Memory.

Combines two mechanisms that complemented each other across datasets:

1. Confusion-matrix-driven disambiguation phase (from confusion_disambiguation_memory):
   Before filling the context budget, inject examples of the labels most frequently
   confused with the top similarity-matched candidates. This anchors the hard label
   boundaries the model has already failed on.

2. MMR fill phase (from mmr_memory):
   After the disambiguation examples are placed, fill the remaining budget with
   Maximum Marginal Relevance selection rather than pure similarity ranking.
   The redundancy penalty prevents near-duplicate examples from piling up around
   the query topic, recovering the diversity that similarity-rank fill loses.

Why this combination should work:
- confusion_disambiguation_memory won avg_val but hurt LawBench because its
  similarity-rank fill clustered redundant examples after the disambiguation phase.
- mmr_memory held the LawBench frontier precisely by suppressing that redundancy.
- Replacing the fill strategy is a targeted fix that should preserve the
  Symptom2Disease gain while recovering LawBench performance.

Cold start: no confusion data → disambiguation phase is skipped → pure MMR fill.
No errors yet: same graceful fallback.
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
_TOP_CANDIDATE_LABELS = 3
_DISAMBIG_PER_CONFUSION = 2
_MMR_LAMBDA = 0.6   # balance relevance vs diversity in fill phase


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class ConfusionMmrMemory(MemorySystem):
    """Confusion-matrix disambiguation phase followed by MMR-diverse fill."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        # confusion[predicted][actual] = count
        self.confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scored(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize(query)
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _confusion_targets_for(self, labels: list[str]) -> list[str]:
        label_set = set(labels)
        counts: dict[str, int] = defaultdict(int)
        for lbl in labels:
            for actual, cnt in self.confusion.get(lbl, {}).items():
                if actual not in label_set:
                    counts[actual] += cnt
        return [lbl for lbl, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        q_tok = _tokenize(query)
        ranked = self._scored(query)

        parts: list[str] = []
        used_indices: set[int] = set()
        total_chars = 0

        def try_add(idx: int, ex: dict) -> bool:
            nonlocal total_chars
            if idx in used_indices:
                return False
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > MAX_CHARS:
                return False
            parts.append(part)
            used_indices.add(idx)
            total_chars += len(part) + 2
            return True

        # Phase 1: confusion-matrix disambiguation examples
        top_labels = [ex["target"] for _, _, ex in ranked[:_TOP_CANDIDATE_LABELS]]
        confused_with = self._confusion_targets_for(top_labels)
        if confused_with:
            disambig_labels_needed = set(confused_with[:_TOP_CANDIDATE_LABELS])
            per_label_disambig: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
            for score, idx, ex in ranked:
                if ex["target"] in disambig_labels_needed:
                    per_label_disambig[ex["target"]].append((score, idx, ex))

            for _round in range(_DISAMBIG_PER_CONFUSION):
                for lbl in confused_with[:_TOP_CANDIDATE_LABELS]:
                    for score, idx, ex in per_label_disambig.get(lbl, []):
                        if idx not in used_indices:
                            try_add(idx, ex)
                            break

        # Phase 2: MMR fill for remaining budget
        # Candidates are all ranked examples not yet used
        candidates = [(s, i, ex) for s, i, ex in ranked if i not in used_indices]
        # Already-selected token sets for redundancy measurement
        selected_tokens = [_tokenize(ex.get("raw_question", ex["input"])) for ex in
                           [self.examples[int(p.split("A: ")[-1]
                                            .split("\n")[0])] if False else ex
                            for ex in []]]
        # Build token sets from parts already added
        selected_token_sets: list[frozenset[str]] = []
        for p in parts:
            q_text = p.split("\nA:")[0][3:]  # strip "Q: " prefix
            selected_token_sets.append(_tokenize(q_text))

        while candidates and total_chars < MAX_CHARS:
            best_score = -999.0
            best_item: tuple[float, int, dict] | None = None

            for s, i, ex in candidates:
                q = ex.get("raw_question", ex["input"])
                part = f"Q: {q}\nA: {ex['target']}"
                if total_chars + len(part) + 2 > MAX_CHARS:
                    continue
                tok = ex["tokens"]
                redundancy = max(
                    (_jaccard(tok, st) for st in selected_token_sets),
                    default=0.0,
                )
                mmr_score = _MMR_LAMBDA * s - (1.0 - _MMR_LAMBDA) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = (s, i, ex)

            if best_item is None:
                break
            _, idx, ex = best_item
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            parts.append(part)
            used_indices.add(idx)
            total_chars += len(part) + 2
            selected_token_sets.append(ex["tokens"])
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
            "num_confusion_pairs": sum(len(v) for v in self.confusion.values()),
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

            if not r.get("was_correct", True):
                pred = r.get("prediction", "")
                gt = r["ground_truth"]
                if pred and pred != gt:
                    self.confusion[pred][gt] += 1

    def get_context_length(self) -> int:
        return sum(len(p) + 2 for p in self._build_parts(""))

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps({"examples": serialisable, "confusion": confusion_plain}, indent=2)

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

        self.confusion = defaultdict(lambda: defaultdict(int))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = cnt
