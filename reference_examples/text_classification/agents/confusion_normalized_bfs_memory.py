"""Confusion-Normalized BFS Memory.

Builds on adaptive_tokenizer_confusion_memory with two targeted fixes:

1. Prediction normalization bug fix: the base system stores raw predictions as
   confusion keys (e.g. '[罪名]诈骗<eoa>') but looks them up using clean
   ground-truth labels ('诈骗'). This means the disambiguation phase never fires —
   zero overlap between keys and lookup labels. This system normalizes predictions
   before storing, so all 133 confusion edges become active.

2. BFS depth-2 traversal: after normalizing, traverse the confusion graph up to
   depth 2 with a (count / depth) weight decay so that direct confusors rank higher
   than transitive ones. This surfaces label clusters the single-hop lookup misses.

Retrieval algorithm:
1. Score all stored examples with adaptive Jaccard against the query.
2. Extract top-3 candidate labels from highest-scoring examples.
3. BFS from those labels over the normalized confusion graph (depth ≤ 2).
4. Inject one disambiguation example per BFS neighbor, ordered by weighted score.
5. Fill remaining budget with similarity-ranked examples.
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
_TOP_CANDIDATE_LABELS = 3
_BFS_MAX_DEPTH = 2
_BFS_MAX_NODES = 8

# CJK Unicode ranges: CJK Unified, Hiragana, Katakana, Fullwidth/Halfwidth
_CJK_RE = re.compile(r'[一-鿿぀-ゟ゠-ヿ＀-￯]')

# Format wrapper patterns added by various prompt templates, e.g. [TAG]answer<eoa>
_WRAP_PREFIX = re.compile(r'^\[[^\]]+\]')
_WRAP_SUFFIX_EOA = re.compile(r'<eoa>$')
_WRAP_SUFFIX_CLOSE = re.compile(r'\[/[^\]]+\]$')


def _normalize_label(s: str) -> str:
    """Strip common output-format wrappers so confusion keys match ground-truth labels."""
    s = s.strip()
    s = _WRAP_PREFIX.sub('', s)
    s = _WRAP_SUFFIX_EOA.sub('', s)
    s = _WRAP_PREFIX.sub('', s)          # second pass for nested wrappers
    s = _WRAP_SUFFIX_CLOSE.sub('', s)
    return s.strip()


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


class ConfusionNormalizedBfsMemory(MemorySystem):
    """Confusion-matrix retrieval with normalized keys and BFS depth-2 traversal.

    Fixes the key mismatch bug in prior systems: raw predictions contain format
    wrappers (e.g. '[TAG]label<eoa>') that never match clean ground-truth labels,
    so the disambiguation phase never fired. Normalizing before storing unlocks
    all accumulated confusion edges.

    BFS depth-2 traversal with (count / depth) weight decay surfaces label clusters
    connected through transitive confusion chains, not just direct one-hop neighbors.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        # confusion[pred_normalized][gt] = count  — both keys are clean labels
        self.confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _scored(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _bfs_confusion_targets(self, seed_labels: list[str]) -> list[str]:
        """BFS over the normalized confusion graph.

        Returns neighboring labels ordered by (count / depth), so direct confusors
        rank above transitive ones. Seeds are excluded from the result.
        """
        seed_set = set(seed_labels)
        visited = set(seed_labels)
        scored: dict[str, float] = {}
        queue = deque((lbl, 0) for lbl in seed_labels)

        while queue:
            node, depth = queue.popleft()
            if depth >= _BFS_MAX_DEPTH:
                continue
            for neighbor, cnt in self.confusion.get(node, {}).items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    # weight decays with depth so depth-1 neighbors stay on top
                    scored[neighbor] = scored.get(neighbor, 0.0) + cnt / (depth + 1)
                    if len(visited) < _BFS_MAX_NODES + len(seed_set):
                        queue.append((neighbor, depth + 1))

        return [lbl for lbl, _ in sorted(scored.items(), key=lambda x: (-x[1], x[0]))]

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        ranked = self._scored(query)
        top_labels = [ex["target"] for _, _, ex in ranked[:_TOP_CANDIDATE_LABELS]]
        confused_with = self._bfs_confusion_targets(top_labels)

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
                # Normalize the prediction before storing so keys match ground-truth labels
                pred_raw = r.get("prediction", "")
                pred = _normalize_label(pred_raw)
                gt = r["ground_truth"]
                if pred and pred != gt:
                    self.confusion[pred][gt] += 1

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
        self.confusion = defaultdict(lambda: defaultdict(int))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = cnt
