"""Recent-Error Hint Memory.

Extends decayed_confusion_memory with a new prompt architecture (axis A): a compact
"Recent corrections" section prepended above the retrieved examples.

Problem with the frontier: the model only sees what's wrong indirectly via retrieved
examples whose labels happened to be confused in the past. There is no direct signal
in the prompt saying "you have been making this specific mistake recently." The model
must infer the current failure pattern from the distribution of retrieved examples,
which is a weak signal when the confusion involves similar-looking inputs.

Fix: maintain a rolling window of the last HINT_WINDOW incorrect (pred → gt) pairs.
After each batch, append new errors (deduped) to the window with recency ordering
(most recent first). At predict time, render this as a compact "Recent corrections"
section at the top of the prompt. The section uses a direct ✗ Predicted / ✓ Correct
format to give the model explicit meta-signal about its current failure patterns
before it sees any retrieved examples.

This is orthogonal to retrieval: the same similarity-ranked + confusion-disambiguated
examples are still injected, but the model now also knows explicitly which label pairs
it has been confusing most recently. The hint section is capped to avoid consuming
too much of the context budget.

Axes changed vs frontier: A (prompt template — new hint section), E (learning trigger —
error window updated after each batch).
"""

import json
import re
from collections import defaultdict, deque
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

_HINT_HEADER = "**Recent corrections (pay close attention):**"
_HINT_ITEM = "✗ Predicted: {pred}  →  ✓ Correct: {gt}"

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.
{hint_section}
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
HINT_WINDOW = 10       # max distinct (pred, gt) pairs to show
HINT_MAX_CHARS = 800   # hard cap on hint section size

DECAY_FACTOR = 0.95
PRUNE_THRESHOLD = 0.05

_CJK_RE = re.compile(r'[一-鿿぀-ゟ゠-ヿ＀-￯]')
_WRAPPER_RE = re.compile(r'^\[[^\]]*\]|<eoa>\s*$|\[[^\]]*\]\s*$', re.IGNORECASE)


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
    if not pred:
        return pred
    cleaned = _WRAPPER_RE.sub('', pred).strip()
    return cleaned if cleaned else pred


class RecentErrorHintMemory(MemorySystem):
    """Confusion-matrix retrieval with an explicit recent-error hint section.

    The hint section lists the most recent distinct (wrong_pred → correct_gt) pairs
    at the top of every prompt, giving the model direct meta-signal about its current
    failure patterns on top of the normal example-retrieval context.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.confusion: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Deque of (pred_clean, gt) tuples, most-recent appended last
        self._error_window: deque[tuple[str, str]] = deque(maxlen=HINT_WINDOW * 4)

    # ------------------------------------------------------------------
    # Hint section builder
    # ------------------------------------------------------------------

    def _build_hint_section(self) -> str:
        if not self._error_window:
            return ""
        seen: set[tuple[str, str]] = set()
        lines = [_HINT_HEADER]
        chars = len(_HINT_HEADER) + 1
        # Iterate most-recent first
        for pred, gt in reversed(list(self._error_window)):
            key = (pred, gt)
            if key in seen:
                continue
            seen.add(key)
            item = _HINT_ITEM.format(pred=pred, gt=gt)
            if chars + len(item) + 1 > HINT_MAX_CHARS:
                break
            lines.append(item)
            chars += len(item) + 1
            if len(seen) >= HINT_WINDOW:
                break
        if len(lines) == 1:
            return ""
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Retrieval (same as decayed_confusion_memory)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # MemorySystem interface
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        hint = self._build_hint_section()
        parts = self._build_parts(input)
        examples_section = "\n\n".join(parts)
        hint_block = f"\n{hint}\n" if hint else "\n"
        prompt = PROMPT_TEMPLATE.format(
            hint_section=hint_block,
            examples_section=examples_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_selected": len(parts),
            "hint_pairs": len(set(self._error_window)),
        }

    def _decay_confusion(self) -> None:
        to_del_outer = []
        for pred, actuals in self.confusion.items():
            to_del_inner = [a for a in actuals if actuals[a] * DECAY_FACTOR < PRUNE_THRESHOLD]
            for a in to_del_inner:
                del actuals[a]
            for a in actuals:
                actuals[a] *= DECAY_FACTOR
            if not actuals:
                to_del_outer.append(pred)
        for pred in to_del_outer:
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
                pred_clean = _normalize_prediction(r.get("prediction", ""))
                gt = r["ground_truth"]
                if pred_clean and pred_clean != gt:
                    self.confusion[pred_clean][gt] += 1.0
                    self._error_window.append((pred_clean, gt))

        self._decay_confusion()

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps({
            "examples": serialisable,
            "confusion": confusion_plain,
            "error_window": list(self._error_window),
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
                restored["tokens"] = _tokenize_adaptive(raw_q)
            self.examples.append(restored)
        self.confusion = defaultdict(lambda: defaultdict(float))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = float(cnt)
        self._error_window = deque(maxlen=HINT_WINDOW * 4)
        for pair in data.get("error_window", []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                self._error_window.append((pair[0], pair[1]))
