"""Missed-Component Boost Memory.

Extends decayed_confusion_memory with a second orthogonal learning signal:
per-atomic-component miss counts.

Problem with the frontier: the confusion matrix treats compound label strings as
opaque keys (e.g. "诈骗;组织、领导传销活动" is one key). When the model predicts
"诈骗" instead, the confusion entry is pred="诈骗" → gt="诈骗;组织、领导传销活动",
but the matrix cannot express which *components* were missed. As a result, retrieval
never specifically surfaces examples of "组织、领导传销活动" to help the model learn
that this label commonly co-occurs and is being under-predicted.

Fix: after every incorrect prediction, split both the predicted label and the
ground-truth label on compound delimiters (semicolons, Chinese semicolons). Any
atomic component in the ground truth that was absent from the prediction is a "missed
component" and its count is incremented. At retrieval time, each stored example
receives a bonus proportional to log(1 + sum_of_miss_counts_for_its_components).
This bonus is added on top of the base Jaccard score so that examples whose labels
are chronically missed rise in rank even when their textual similarity is moderate.

Both confusion counts and missed-component counts decay after each batch (same
DECAY_FACTOR as the frontier) so stale signal doesn't dominate indefinitely.

Axes changed vs frontier: B (memory content — new missed-component counter) and
C (selection algorithm — component-overlap bonus in scoring).
"""

import json
import math
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

DECAY_FACTOR = 0.95
PRUNE_THRESHOLD = 0.05
COMPONENT_ALPHA = 0.3   # weight of log-scaled missed-component bonus

_CJK_RE = re.compile(r'[一-鿿぀-ゟ゠-ヿ＀-￯]')
_WRAPPER_RE = re.compile(r'^\[[^\]]*\]|<eoa>\s*$|\[[^\]]*\]\s*$', re.IGNORECASE)
_COMPOUND_SPLIT_RE = re.compile(r'[;；]')


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


def _split_compound(label: str) -> list[str]:
    """Split a (possibly compound) label into its atomic components."""
    parts = [p.strip() for p in _COMPOUND_SPLIT_RE.split(label) if p.strip()]
    return parts if parts else [label]


def _missed_components(pred: str, gt: str) -> list[str]:
    """Return atomic components of gt not present in pred."""
    pred_parts = set(_split_compound(pred))
    gt_parts = set(_split_compound(gt))
    return list(gt_parts - pred_parts)


class MissedComponentMemory(MemorySystem):
    """Confusion + missed-component boost memory.

    Adds a second learning signal on top of decayed_confusion_memory: a per-atomic-
    component miss counter. When a prediction omits part of a compound ground-truth
    label, each missing component's count is incremented. At retrieval time every
    stored example receives a log-scaled bonus proportional to the sum of miss counts
    for its own label components, pulling chronically-missed labels up in rank.
    Both confusion counts and missed-component counts decay after each batch.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.confusion: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # missed_counts[component] = float count of times this component was absent from prediction
        self.missed_counts: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _component_overlap(self, target: str) -> float:
        """Sum of missed_counts for each atomic component of target."""
        return sum(self.missed_counts.get(c, 0.0) for c in _split_compound(target))

    def _scored(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = []
        for idx, ex in enumerate(self.examples):
            sim = _jaccard(q_tok, ex["tokens"])
            bonus = COMPONENT_ALPHA * math.log(1.0 + self._component_overlap(ex["target"]))
            result.append((sim + bonus, idx, ex))
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

        # Disambiguation phase (same as frontier)
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

        # Fill phase — ranked now includes component-overlap bonus
        for _, idx, ex in ranked:
            if total_chars >= MAX_CHARS:
                break
            try_add(idx, ex)

        return parts

    # ------------------------------------------------------------------
    # MemorySystem interface
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = self._build_parts(input)
        examples_section = "\n\n".join(parts)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_selected": len(parts),
            "num_missed_components": len(self.missed_counts),
        }

    def _decay(self) -> None:
        # Decay confusion matrix
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
        # Decay missed counts
        to_del = [c for c, v in self.missed_counts.items() if v * DECAY_FACTOR < PRUNE_THRESHOLD]
        for c in to_del:
            del self.missed_counts[c]
        for c in list(self.missed_counts):
            self.missed_counts[c] *= DECAY_FACTOR

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
                pred_raw = _normalize_prediction(r.get("prediction", ""))
                gt = r["ground_truth"]
                if pred_raw and pred_raw != gt:
                    # Confusion matrix entry
                    self.confusion[pred_raw][gt] += 1.0
                    # Missed-component tracking
                    for component in _missed_components(pred_raw, gt):
                        self.missed_counts[component] += 1.0

        self._decay()

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps({
            "examples": serialisable,
            "confusion": confusion_plain,
            "missed_counts": dict(self.missed_counts),
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
        self.missed_counts = defaultdict(float)
        for comp, cnt in data.get("missed_counts", {}).items():
            self.missed_counts[comp] = float(cnt)
