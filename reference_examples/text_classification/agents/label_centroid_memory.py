"""Label Centroid Routing Memory.

Replaces the confusion-matrix disambiguation phase from
adaptive_tokenizer_confusion_memory with per-label token centroid routing.

Problem with confusion-based systems: the disambiguation phase only fires when a
stored confusion key exactly matches the (normalised) prediction — a brittle lookup
that misses all queries for which the model is uncertain but hasn't been wrong on
that exact pair before.  Confusion counts also accumulate unboundedly on early-
training errors, crowding out signal from later corrections (addressed partially by
decay, but the lookup key mismatch remains).

Fix: maintain a per-label token-frequency centroid.  At predict time, compute
Jaccard similarity between the query and each label's centroid (tokens present in
≥30 % of that label's training examples) to identify the top CENTROID_TOP_LABELS
most likely labels.  Retrieve examples from those candidate labels first (similarity
ranked), then fill the remainder of the context budget globally.  The centroid
routing fires on every query regardless of past errors and requires no exact-match
normalisation.

Selection axis C (algorithm) + memory-content axis B.
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
_CENTROID_TOP_LABELS = 3   # candidate labels returned by centroid routing
_CENTROID_THRESHOLD = 0.3  # token must appear in this fraction of label examples

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
    return _tokenize_bigram(text) if _has_cjk(text) else _tokenize_word(text)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class _LabelCentroid:
    """Running token-frequency centroid for one label."""

    __slots__ = ("_counts", "_n")

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._n: int = 0

    def update(self, tokens: frozenset) -> None:
        self._n += 1
        for t in tokens:
            self._counts[t] += 1

    def representative(self, threshold: float = _CENTROID_THRESHOLD) -> frozenset:
        """Tokens present in at least `threshold` fraction of seen examples."""
        if self._n == 0:
            return frozenset()
        cutoff = self._n * threshold
        return frozenset(t for t, c in self._counts.items() if c >= cutoff)

    def to_dict(self) -> dict:
        return {"counts": dict(self._counts), "n": self._n}

    @classmethod
    def from_dict(cls, d: dict) -> "_LabelCentroid":
        obj = cls()
        obj._counts = defaultdict(int, d.get("counts", {}))
        obj._n = d.get("n", 0)
        return obj


class LabelCentroidMemory(MemorySystem):
    """Per-label centroid routing memory.

    Maintains a token-frequency centroid for every observed label.  At predict
    time, the query is scored against each centroid (Jaccard over the
    high-frequency-token set) to identify the CENTROID_TOP_LABELS most likely
    candidate labels.  Examples from those labels are retrieved first (ranked
    by per-example Jaccard), then the remaining context budget is filled from
    the full pool.  Exponential decay on centroids is not used — the centroid
    threshold already down-weights tokens from early, low-count episodes because
    they don't yet reach the required fraction.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self._centroids: dict[str, _LabelCentroid] = {}

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _top_labels(self, q_tok: frozenset) -> list[str]:
        """Return top CENTROID_TOP_LABELS labels by centroid similarity."""
        scores = {
            lbl: _jaccard(q_tok, centroid.representative())
            for lbl, centroid in self._centroids.items()
        }
        return sorted(scores, key=lambda l: (-scores[l], l))[:_CENTROID_TOP_LABELS]

    def _scored(self, q_tok: frozenset) -> list[tuple[float, int, dict]]:
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        q_tok = _tokenize_adaptive(query)
        candidate_labels = set(self._top_labels(q_tok))

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

        ranked = self._scored(q_tok)

        # Phase 1: centroid-routed fill — examples from top candidate labels first
        for score, idx, ex in ranked:
            if total_chars >= MAX_CHARS:
                break
            if ex["target"] in candidate_labels:
                try_add(idx, ex)

        # Phase 2: fill remainder from full pool (global similarity)
        for score, idx, ex in ranked:
            if total_chars >= MAX_CHARS:
                break
            try_add(idx, ex)

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
            "num_labels": len(self._centroids),
            "num_selected": len(parts),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            toks = _tokenize_adaptive(raw_q)
            ex: dict[str, Any] = {
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": toks,
            }
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

            # Update centroid for this label
            lbl = r["ground_truth"]
            if lbl not in self._centroids:
                self._centroids[lbl] = _LabelCentroid()
            self._centroids[lbl].update(toks)

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        centroid_data = {lbl: c.to_dict() for lbl, c in self._centroids.items()}
        return json.dumps({"examples": serialisable, "centroids": centroid_data}, indent=2)

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
        self._centroids = {
            lbl: _LabelCentroid.from_dict(d)
            for lbl, d in data.get("centroids", {}).items()
        }
