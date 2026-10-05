"""Anchor-Recency Confusion Memory.

Addresses the unbounded-pool problem of the frontier: as training progresses, the
flat pool grows to hundreds of examples, but context can only fit ~135 of them.
Late-arriving labels end up under-represented because early examples dominate by
sheer count.

Memory structure:
- anchors: one (most recently seen) example per label — guarantees every observed
  label is always in the pool, regardless of pool size.
- recency: a bounded sliding window of the last RECENCY_WINDOW examples — keeps
  recent difficulty patterns fresh.
- confusion: standard confusion matrix (normalized predictions as keys).

Retrieval:
1. Build the candidate pool as union(anchors.values(), recency), deduplicated.
2. Score all pool examples with adaptive Jaccard against the query.
3. Identify confusion targets from top-3 candidate labels (same as frontier).
4. Inject one disambiguation example per confusion target (up to DISAMBIG_SLOTS).
5. Fill remaining budget with similarity-ranked pool examples.

Why this is different from the frontier:
- Pool size is O(num_labels + RECENCY_WINDOW) rather than O(training_steps).
- Every label seen during training is guaranteed one slot in the pool via anchors.
- The recency window provides temporal freshness without drowning anchors.
"""

import json
import re
from collections import defaultdict, deque
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
RECENCY_WINDOW = 80
_TOP_CANDIDATE_LABELS = 3
_DISAMBIG_SLOTS = 3

_CJK_RE = re.compile(r'[一-鿿぀-ゟ゠-ヿ＀-￯]')


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
    """Route to bigrams for CJK text, word tokens for ASCII/Latin."""
    return _tokenize_bigram(text) if _has_cjk(text) else _tokenize_word(text)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class AnchorRecencyConfusionMemory(MemorySystem):
    """Bounded anchor+recency pool with confusion disambiguation.

    The anchor dict guarantees full label coverage; the recency window captures
    recent difficulty patterns. Pool size stays O(labels + RECENCY_WINDOW)
    instead of growing unboundedly with training steps.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        # label -> most recently seen example for that label
        self.anchors: dict[str, dict[str, Any]] = {}
        # bounded sliding window of recent examples
        self.recency: deque[dict[str, Any]] = deque(maxlen=RECENCY_WINDOW)
        # confusion[predicted_label][actual_label] = count
        self.confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _build_pool(self) -> list[dict[str, Any]]:
        """Union of anchor examples and recency window, deduplicated by object id."""
        seen: set[int] = set()
        pool: list[dict[str, Any]] = []
        for ex in self.recency:
            if id(ex) not in seen:
                seen.add(id(ex))
                pool.append(ex)
        for ex in self.anchors.values():
            if id(ex) not in seen:
                seen.add(id(ex))
                pool.append(ex)
        return pool

    def _scored(self, query: str, pool: list[dict[str, Any]]) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(pool)
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
        pool = self._build_pool()
        if not pool:
            return []

        ranked = self._scored(query, pool)
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

        # Phase 1: disambiguation — one best example per confusion target
        if confused_with:
            per_label: dict[str, int | None] = {
                lbl: None for lbl in confused_with[:_DISAMBIG_SLOTS]
            }
            disambig_set = set(per_label.keys())
            for _, idx, ex in ranked:
                if ex["target"] in disambig_set and per_label[ex["target"]] is None:
                    per_label[ex["target"]] = idx
                if all(v is not None for v in per_label.values()):
                    break
            for lbl in confused_with[:_DISAMBIG_SLOTS]:
                if per_label.get(lbl) is not None:
                    try_add(per_label[lbl], pool[per_label[lbl]])

        # Phase 2: similarity-ranked fill
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
            "num_anchors": len(self.anchors),
            "num_recency": len(self.recency),
            "num_selected": len(parts),
            "num_confusion_pairs": sum(len(v) for v in self.confusion.values()),
        }

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

            # Update anchor (always keep most recent per label)
            self.anchors[r["ground_truth"]] = ex
            # Append to recency window
            self.recency.append(ex)

            # Update confusion matrix (normalize prediction to strip format wrappers)
            if not r.get("was_correct", True):
                pred = r.get("prediction", "")
                gt = r["ground_truth"]
                if pred and pred != gt:
                    self.confusion[pred][gt] += 1

    def get_state(self) -> str:
        def serialise_ex(ex: dict) -> dict:
            return {
                k: (sorted(v) if isinstance(v, frozenset) else v)
                for k, v in ex.items()
            }

        anchors_ser = {lbl: serialise_ex(ex) for lbl, ex in self.anchors.items()}
        recency_ser = [serialise_ex(ex) for ex in self.recency]
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps(
            {"anchors": anchors_ser, "recency": recency_ser, "confusion": confusion_plain},
            indent=2,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)

        def restore_ex(ex: dict) -> dict:
            restored = dict(ex)
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                raw_q = restored.get("raw_question", restored.get("input", ""))
                restored["tokens"] = _tokenize_adaptive(raw_q)
            return restored

        self.anchors = {
            lbl: restore_ex(ex) for lbl, ex in data.get("anchors", {}).items()
        }
        self.recency = deque(
            (restore_ex(ex) for ex in data.get("recency", [])),
            maxlen=RECENCY_WINDOW,
        )
        self.confusion = defaultdict(lambda: defaultdict(int))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = cnt
