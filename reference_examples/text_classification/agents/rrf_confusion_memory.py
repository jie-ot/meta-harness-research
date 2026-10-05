"""Reciprocal Rank Fusion Confusion Memory.

Replaces the two-phase confusion-slot/similarity-fill retrieval in
decayed_confusion_memory with a single parameter-free RRF pass.

Problem with the two-phase design: confusion slots are reserved unconditionally
for the top confusion targets regardless of whether those examples are also
topically relevant to the current query.  When the confusion targets happen to
be unrelated to the query, they consume context budget with low-utility examples
and displace higher-utility similarity hits.  unified_confusion_retrieval_memory
attempted a linear bonus (alpha * log(1 + affinity)) but required a tuned alpha
and still regressed.

Fix: compute two independent ranked lists —
  (1) examples ranked by adaptive Jaccard similarity to the query,
  (2) examples ranked by the column affinity of their label in the confusion
      matrix (i.e. how often that label is the *true* label when the model is
      wrong), so examples of frequently-missed labels rank higher.

Fuse with Reciprocal Rank Fusion (k=60): score_i = 1/(k + sim_rank_i) + 1/(k + aff_rank_i).
Select greedily by descending RRF score until the context budget is full.  No
alpha to tune; the two signals contribute symmetrically.

At cold start (empty confusion matrix) all column affinities are zero, so
aff_rank is arbitrary-but-stable and the fused order degrades gracefully to
pure similarity order.  Decay and prediction normalisation are inherited
unchanged from decayed_confusion_memory.
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
RRF_K = 60               # RRF smoothing constant (standard value)

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


class RrfConfusionMemory(MemorySystem):
    """Confusion memory with Reciprocal Rank Fusion retrieval.

    Fuses two ranked lists — (1) adaptive Jaccard similarity and (2) confusion
    column affinity (how often each label is the true label when the model errs)
    — using parameter-free RRF scoring: 1/(k+sim_rank) + 1/(k+aff_rank).
    Replaces the two-phase slot-reservation design entirely.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.confusion: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _col_affinity(self) -> dict[str, float]:
        """Column sums of the confusion matrix: how often each label is the true label."""
        aff: dict[str, float] = defaultdict(float)
        for actuals in self.confusion.values():
            for actual, cnt in actuals.items():
                aff[actual] += cnt
        return aff

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        q_tok = _tokenize_adaptive(query)
        n = len(self.examples)

        # Rank 1: by similarity (descending)
        sim_order = sorted(range(n), key=lambda i: -_jaccard(q_tok, self.examples[i]["tokens"]))
        sim_rank = [0] * n
        for r, i in enumerate(sim_order):
            sim_rank[i] = r

        # Rank 2: by column affinity of the example's label (descending)
        aff = self._col_affinity()
        aff_order = sorted(range(n), key=lambda i: -aff.get(self.examples[i]["target"], 0.0))
        aff_rank = [0] * n
        for r, i in enumerate(aff_order):
            aff_rank[i] = r

        # Fuse with RRF
        rrf_order = sorted(
            range(n),
            key=lambda i: -(1.0 / (RRF_K + sim_rank[i]) + 1.0 / (RRF_K + aff_rank[i])),
        )

        parts: list[str] = []
        total_chars = 0
        for idx in rrf_order:
            ex = self.examples[idx]
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > MAX_CHARS:
                break
            parts.append(part)
            total_chars += len(part) + 2

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
            "num_confusion_entries": sum(len(v) for v in self.confusion.values()),
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
                    self.confusion[pred][gt] += 1.0

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
