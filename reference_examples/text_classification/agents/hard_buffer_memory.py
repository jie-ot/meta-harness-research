"""Hard-Example Buffer Memory.

Augments the decayed_confusion_memory general pool with a structurally separate
bounded FIFO buffer of error examples.

Problem with confusion-based systems: the confusion matrix is a sparse count over
label pairs — it records *which* labels are confused but stores no retrievable text
for the confused cases.  Examples of confusable inputs are sitting in the general
pool but are retrieved only if they happen to be similar to the current query; there
is no mechanism that preferentially surfaces examples that the model historically
mis-predicted.

Fix: every example where was_correct=False is also appended to a bounded hard
buffer (max HARD_BUFFER_SIZE entries, FIFO eviction).  At predict time, score
hard-buffer examples against the query with adaptive Jaccard and inject the top
HARD_SLOTS examples that exceed HARD_SIM_THRESHOLD similarity before filling the
rest of the context from the general pool.  The similarity gate prevents injecting
unrelated error examples that would dilute context quality.

This differs from utility_weighted_memory (which applied per-example outcome scores
to general-pool retrieval) because errors are held in a physically separate pool with
independent eviction; it differs from contrastive_error_memory (which stored explicit
wrong/correct pairs) because the prompt structure is unchanged — only the retrieval
source differs.

Memory-structure axis D + selection-algorithm axis C.
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
_HARD_BUFFER_SIZE = 60      # maximum error examples retained (FIFO eviction)
_HARD_SLOTS = 3             # max hard-buffer examples injected per query
_HARD_SIM_THRESHOLD = 0.10  # minimum Jaccard similarity to inject a hard example
_TOP_CANDIDATE_LABELS = 3   # top retrieved labels used to query confusion matrix
_DISAMBIG_PER_CONFUSION = 2 # rounds of confusion-slot injection
DECAY_FACTOR = 0.95         # multiply confusion counts after each batch
_DECAY_FACTOR = DECAY_FACTOR
_PRUNE_THRESHOLD = 0.05     # drop confusion entries below this count

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
    if not pred:
        return pred
    cleaned = _WRAPPER_RE.sub('', pred).strip()
    return cleaned if cleaned else pred


class HardBufferMemory(MemorySystem):
    """Decayed confusion memory augmented with a structurally separate hard-example buffer.

    At predict time, the hard buffer is scored against the query and its top
    HARD_SLOTS most-similar examples (above HARD_SIM_THRESHOLD) are injected
    before the general similarity fill.  This ensures examples from error-
    producing inputs are preferentially surfaced when a new query resembles them,
    without the fragile exact-key lookup that confusion-matrix systems require.
    The confusion matrix is retained for additional disambiguation slots.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        # Bounded hard buffer: examples where was_correct=False (FIFO)
        self._hard: list[dict[str, Any]] = []
        self.confusion: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _scored(self, query: str, pool: list) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(pool)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _confusion_targets_for(self, labels: list[str]) -> list[str]:
        label_set = set(labels)
        counts: dict[str, float] = defaultdict(float)
        for lbl in labels:
            for actual, cnt in self.confusion.get(lbl, {}).items():
                if actual not in label_set:
                    counts[actual] += cnt
        return [lbl for lbl, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        q_tok = _tokenize_adaptive(query)
        parts: list[str] = []
        used_ids: set[int] = set()
        total_chars = 0

        def try_add(ex: dict) -> bool:
            nonlocal total_chars
            eid = id(ex)
            if eid in used_ids:
                return False
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > MAX_CHARS:
                return False
            parts.append(part)
            used_ids.add(eid)
            total_chars += len(part) + 2
            return True

        # Phase 1: similarity-gated hard buffer injection
        hard_scored = sorted(
            self._hard,
            key=lambda e: _jaccard(q_tok, e["tokens"]),
            reverse=True,
        )
        hard_added = 0
        for ex in hard_scored:
            if hard_added >= _HARD_SLOTS:
                break
            sim = _jaccard(q_tok, ex["tokens"])
            if sim < _HARD_SIM_THRESHOLD:
                break
            if try_add(ex):
                hard_added += 1

        # Phase 2: confusion disambiguation (same as baseline)
        ranked = self._scored(query, self.examples)
        top_labels = [ex["target"] for _, _, ex in ranked[:_TOP_CANDIDATE_LABELS]]
        confused_with = self._confusion_targets_for(top_labels)

        if confused_with:
            disambig_labels = set(confused_with[:_TOP_CANDIDATE_LABELS])
            per_label: dict[str, list] = defaultdict(list)
            for score, idx, ex in ranked:
                if ex["target"] in disambig_labels:
                    per_label[ex["target"]].append((score, idx, ex))
            for _round in range(_DISAMBIG_PER_CONFUSION):
                for lbl in confused_with[:_TOP_CANDIDATE_LABELS]:
                    for _, _, ex in per_label.get(lbl, []):
                        if id(ex) not in used_ids:
                            try_add(ex)
                            break

        # Phase 3: fill remaining budget from general pool
        for _, _, ex in ranked:
            if total_chars >= MAX_CHARS:
                break
            try_add(ex)

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
            "num_hard": len(self._hard),
            "num_selected": len(parts),
        }

    def _decay_confusion(self) -> None:
        to_del_outer = []
        for pred, actuals in self.confusion.items():
            to_del_inner = [a for a, c in actuals.items() if c * _DECAY_FACTOR < _PRUNE_THRESHOLD]
            for a in to_del_inner:
                del actuals[a]
            for a in actuals:
                actuals[a] *= _DECAY_FACTOR
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
                # Add to hard buffer (FIFO)
                self._hard.append(ex)
                if len(self._hard) > _HARD_BUFFER_SIZE:
                    self._hard.pop(0)
                # Update confusion matrix
                pred = _normalize_prediction(r.get("prediction", ""))
                gt = r["ground_truth"]
                if pred and pred != gt:
                    self.confusion[pred][gt] += 1.0

        self._decay_confusion()

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        hard_serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self._hard
        ]
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps({
            "examples": serialisable,
            "hard": hard_serialisable,
            "confusion": confusion_plain,
        }, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)

        def restore_examples(lst):
            out = []
            for ex in lst:
                restored = dict(ex)
                if "tokens" in restored and isinstance(restored["tokens"], list):
                    restored["tokens"] = frozenset(restored["tokens"])
                else:
                    raw_q = restored.get("raw_question", restored.get("input", ""))
                    restored["tokens"] = _tokenize_adaptive(raw_q)
                out.append(restored)
            return out

        self.examples = restore_examples(data.get("examples", []))
        self._hard = restore_examples(data.get("hard", []))
        self.confusion = defaultdict(lambda: defaultdict(float))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = float(cnt)
