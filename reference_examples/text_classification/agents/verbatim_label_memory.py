"""Verbatim Label Memory.

Extends symmetric_confusion_memory with a prompt-template change (axis A) that
adds an explicit verbatim-copy and multi-label completeness instruction.

Problem: trace analysis on the frontier system showed two recurring format error
classes that account for roughly a third of all errors on tasks with structured
label spaces:
  1. Label-suffix inflation — the model appends common categorical suffixes (e.g.
     "罪" in Chinese legal tasks) to label strings that don't carry that suffix in
     the training data.
  2. Partial multi-label output — when a ground-truth answer contains several
     semicolon-separated components, the model predicts only the most salient one
     and omits the rest.

Both error classes share the same root: the model is paraphrasing rather than
copying. Neither class requires better retrieval to fix — they persist even when
the correct label appears in context.

Fix: add two sentences to the instruction section of the prompt:
  a) Copy answer labels character-for-character from the examples — do not
     paraphrase, add suffixes, or alter the phrasing.
  b) If the answer requires multiple labels, include every label separated by
     semicolons just as shown in the examples.

All retrieval mechanics (adaptive tokenizer, two-phase confusion/similarity fill,
prediction normalization, symmetric confusion edges, decay) are unchanged.
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
- Copy answer labels character-for-character from the examples — do not paraphrase, add suffixes, or alter the phrasing in any way
- If the answer requires multiple labels, include every label separated by semicolons exactly as shown in the examples
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000
_TOP_CANDIDATE_LABELS = 3
_DISAMBIG_PER_CONFUSION = 2

DECAY_FACTOR = 0.95      # multiply all counts after each step
PRUNE_THRESHOLD = 0.05   # drop entries with count below this

_CJK_RE = re.compile(r'[一-鿿぀-ゟ゠-ヿ＀-￯]')
_WRAPPER_RE = re.compile(r'^\s*\[[^\]]*\]|<eoa>\s*$|\[[^\]]*\]\s*$', re.IGNORECASE)


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _tokenize_word(text: str) -> frozenset:
    return frozenset(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _tokenize_bigram(text: str) -> frozenset:
    t = re.sub(r'\s+', ' ', text.strip())
    if len(t) < 2:
        return frozenset([t]) if t else frozenset()
    return frozenset(t[i:i + 2] for i in range(len(t) - 1))


def _tokenize_adaptive(text: str) -> frozenset:
    return _tokenize_bigram(text) if _has_cjk(text) else _tokenize_word(text)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_prediction(pred: str) -> str:
    """Strip format wrappers so predictions match ground-truth label strings."""
    if not pred:
        return pred
    cleaned = _WRAPPER_RE.sub('', pred).strip()
    return cleaned if cleaned else pred


class VerbatimLabelMemory(MemorySystem):
    """Symmetric confusion memory with verbatim-copy and multi-label completeness prompt instructions."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        # Float-valued counts to allow decay
        self.confusion: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _scored(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _confusion_targets_for(self, labels: list[str]) -> list[str]:
        label_set = set(labels)
        counts: dict[str, float] = defaultdict(float)
        for lbl in labels:
            row = self.confusion.get(lbl, {})
            for actual, cnt in row.items():
                if actual not in label_set:
                    counts[actual] += cnt
        return [lbl for lbl, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        ranked = self._scored(query)
        top_labels = [ex["target"] for _, _, ex in ranked[:_TOP_CANDIDATE_LABELS]]
        confused_with = self._confusion_targets_for(top_labels)

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

        if confused_with:
            disambig_labels_needed = set(confused_with[:_TOP_CANDIDATE_LABELS])
            per_label_disambig: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
            for score, idx, ex in ranked:
                if ex["target"] in disambig_labels_needed:
                    per_label_disambig[ex["target"]].append((score, idx, ex))

            for _round in range(_DISAMBIG_PER_CONFUSION):
                for lbl in confused_with[:_TOP_CANDIDATE_LABELS]:
                    pool = per_label_disambig.get(lbl, [])
                    for score, idx, ex in pool:
                        if idx not in used_indices:
                            try_add(idx, ex)
                            break

        for _, idx, ex in ranked:
            if total_chars >= MAX_CHARS:
                break
            try_add(idx, ex)

        return parts

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = self._build_parts(input)
        examples_section = "\n\n".join(parts)
        prompt = PROMPT_TEMPLATE.format(
            examples_section=examples_section,
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

    def _decay_confusion(self) -> None:
        """Multiply all confusion counts by DECAY_FACTOR and prune near-zero entries."""
        to_delete_outer = []
        for pred, actuals in self.confusion.items():
            to_delete_inner = []
            for actual in actuals:
                actuals[actual] *= DECAY_FACTOR
                if actuals[actual] < PRUNE_THRESHOLD:
                    to_delete_inner.append(actual)
            for actual in to_delete_inner:
                del actuals[actual]
            if not actuals:
                to_delete_outer.append(pred)
        for pred in to_delete_outer:
            del self.confusion[pred]

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            ex: dict[str, Any] = {
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": _tokenize_adaptive(raw_q),
            }
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

            if not r.get("was_correct", True):
                pred = _normalize_prediction(r.get("prediction", ""))
                gt = r["ground_truth"]
                if pred and pred != gt:
                    # Forward edge: pred was confused with gt
                    self.confusion[pred][gt] += 1.0
                    # Reverse edge at half weight: gt is also confused with pred.
                    # The disambiguation phase looks up stored examples by the top
                    # candidate labels retrieved from the example pool, which are
                    # ground-truth labels — not raw predictions. Without the reverse
                    # edge, those lookups almost never find a match. Adding gt→pred
                    # at 0.5 makes the phase fire for the majority of queries while
                    # keeping forward errors at higher priority.
                    self.confusion[gt][pred] += 0.5

        # Decay after processing the full batch
        self._decay_confusion()

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
                restored["tokens"] = _tokenize_adaptive(raw_q)
            self.examples.append(restored)
        self.confusion = defaultdict(lambda: defaultdict(float))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = float(cnt)
