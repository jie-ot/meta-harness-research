"""Category-Cluster Memory.

A fundamentally different retrieval architecture (axis C — selection algorithm)
that groups stored examples by a structural category tag extracted from the input
and prioritizes within-category examples before falling back to cross-category
similarity.

Problem with all prior systems: they retrieve by Jaccard similarity over the full
input text. When inputs share a long boilerplate preamble (a fixed task description,
domain context, or format specification), that preamble dominates the similarity
score and retrieval becomes near-random with respect to the specific instance.

Fix: extract a short structural tag from the input (e.g., the reaction category,
task type, or domain prefix) using pattern matching, and organize retrieval so that
examples sharing the same tag are always considered first. Within a category bucket,
similarity still ranks examples; the budget remaining after filling from the same
category is filled from the rest of the pool by similarity. This is a two-level
retrieval hierarchy: category first, then similarity within and across buckets.

The key insight from prototyping: the category tag is almost always inferable from
a short fragment of the input (first sentence, a labeled field like "type: X", or
a colon-delimited prefix). Once the category is known, same-category examples are
far more informative than cross-category examples with slightly better token overlap.

Retains the decayed confusion mechanism from the frontier for the disambiguation
phase, since it provides incremental signal without cost when errors occur.
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
_DISAMBIG_PER_CONFUSION = 2
_TOP_CONFUSION_LABELS = 3

DECAY_FACTOR = 0.95
PRUNE_THRESHOLD = 0.05

_CJK_RE = re.compile(r'[一-鿿぀-ゟ゠-ヿ＀-￯]')
_WRAPPER_RE = re.compile(r'^\s*\[[^\]]*\]|<eoa>\s*$|\[[^\]]*\]\s*$', re.IGNORECASE)

# Patterns to extract a short category/type tag from input text.
# Each pattern captures a value that identifies the structural class of the input.
_TAG_PATTERNS = [
    # "the reaction type is X" / "reaction type: X"
    re.compile(r'reaction type is ([^.\n<]+)', re.IGNORECASE),
    # "type: X" or "category: X" or "class: X" or "task: X" on its own segment
    re.compile(r'(?:^|\n|\.)\s*(?:type|category|class|task)[:\s]+([^\n.<]{3,60})', re.IGNORECASE),
    # "Context: X" as a labeled field (first occurrence only)
    re.compile(r'(?:^|\n)Context:\s*([^\n.<]{3,60})', re.IGNORECASE),
    # First sentence ending with a period or newline (capped at 80 chars)
    re.compile(r'^([^.\n]{8,80})[.\n]'),
]
_MAX_TAG_CHARS = 60


def _extract_category_tag(text: str) -> str:
    """Return a lowercased, stripped structural category tag or '' if not found."""
    for pattern in _TAG_PATTERNS:
        m = pattern.search(text)
        if m:
            tag = m.group(1).strip().lower()
            if tag:
                return tag[:_MAX_TAG_CHARS]
    return ''


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


class CategoryClusterMemory(MemorySystem):
    """Two-level retrieval: same-category examples first, then cross-category by similarity.

    Stored examples are indexed both by a structural category tag extracted from
    their input and by Jaccard tokens. At predict time:
      1. Extract the category tag from the query.
      2. Score all same-category examples by Jaccard and fill greedily up to budget.
      3. Fill remaining budget from the rest of the pool, also by Jaccard.
    The confusion disambiguation phase runs on top: before the fill steps, inject
    examples for labels that the decayed confusion matrix says are most confused,
    using same-category preference within those too.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.confusion: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_pool(self, q_tok: frozenset,
                    pool: list[tuple[int, dict]]) -> list[tuple[float, int, dict]]:
        result = [(_jaccard(q_tok, ex["tokens"]), idx, ex) for idx, ex in pool]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _confusion_targets(self, labels: list[str]) -> list[str]:
        counts: dict[str, float] = defaultdict(float)
        label_set = set(labels)
        for lbl in labels:
            for actual, cnt in self.confusion.get(lbl, {}).items():
                if actual not in label_set:
                    counts[actual] += cnt
        return [lbl for lbl, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        tag = _extract_category_tag(query)
        q_tok = _tokenize_adaptive(query)

        # Partition pool into same-category and other
        same: list[tuple[int, dict]] = []
        other: list[tuple[int, dict]] = []
        for idx, ex in enumerate(self.examples):
            (same if ex.get("category_tag") == tag and tag else other).append((idx, ex))

        # When tag is empty or same pool is small, treat everything as one pool
        if not tag or len(same) < 2:
            same = list(enumerate(self.examples))
            other = []

        same_ranked = self._score_pool(q_tok, same)
        other_ranked = self._score_pool(q_tok, other)

        # Interleaved full ranking: same first, then other
        ranked = same_ranked + other_ranked
        top_labels = [ex["target"] for _, _, ex in ranked[:3]]
        confused_with = self._confusion_targets(top_labels)

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

        # Confusion disambiguation slots (same-category preferred)
        if confused_with:
            target_set = set(confused_with[:_TOP_CONFUSION_LABELS])
            per_label_same: dict[str, list] = defaultdict(list)
            per_label_other: dict[str, list] = defaultdict(list)
            for score, idx, ex in same_ranked:
                if ex["target"] in target_set:
                    per_label_same[ex["target"]].append((score, idx, ex))
            for score, idx, ex in other_ranked:
                if ex["target"] in target_set:
                    per_label_other[ex["target"]].append((score, idx, ex))

            for _round in range(_DISAMBIG_PER_CONFUSION):
                for lbl in confused_with[:_TOP_CONFUSION_LABELS]:
                    # Try same-category first, then cross-category
                    for pool_list in (per_label_same.get(lbl, []),
                                      per_label_other.get(lbl, [])):
                        for score, idx, ex in pool_list:
                            if idx not in used_indices:
                                try_add(idx, ex)
                                break
                        else:
                            continue
                        break

        # Fill: same-category first, then cross-category
        for _, idx, ex in same_ranked:
            if total_chars >= MAX_CHARS:
                break
            try_add(idx, ex)

        for _, idx, ex in other_ranked:
            if total_chars >= MAX_CHARS:
                break
            try_add(idx, ex)

        return parts

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = self._build_parts(input)
        examples_section = "\n\n".join(parts) if parts else "(no examples yet)"
        prompt = PROMPT_TEMPLATE.format(
            examples_section=examples_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "query_tag": _extract_category_tag(input),
            "num_examples": len(self.examples),
            "num_selected": len(parts),
            "num_confusion_pairs": sum(len(v) for v in self.confusion.values()),
        }

    def _decay_confusion(self) -> None:
        to_del_outer = []
        for pred, actuals in self.confusion.items():
            to_del_inner = [a for a, c in actuals.items()
                            if c * DECAY_FACTOR < PRUNE_THRESHOLD]
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
                "category_tag": _extract_category_tag(raw_q),
            }
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

            if not r.get("was_correct", True):
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
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps(
            {"examples": serialisable, "confusion": confusion_plain}, indent=2
        )

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
            if "category_tag" not in restored:
                raw_q = restored.get("raw_question", restored.get("input", ""))
                restored["category_tag"] = _extract_category_tag(raw_q)
            self.examples.append(restored)
        self.confusion = defaultdict(lambda: defaultdict(float))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = float(cnt)
