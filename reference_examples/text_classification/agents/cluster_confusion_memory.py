"""Cluster Confusion Memory.

Extends adaptive_tokenizer_confusion_memory with a transitive confusion-cluster
graph. The frontier system only looks at direct confusions (predicted→actual
one hop). This system computes connected components of the confusion graph at
retrieval time and guarantees one representative per *cluster member*, not just
per directly-confused label.

Mechanism:
1. Same adaptive tokenizer as the frontier.
2. During learning, confusion edges are stored as usual: predicted→actual.
3. At predict time, build a graph from all confusion edges (both directions),
   compute the connected component containing any top-3 candidate label, and
   use ALL members of that cluster as the disambiguation target set.
4. Select the most query-similar example for each cluster member (one pass,
   similarity-ranked), then fill remaining budget with regular examples.

The hypothesis: on tasks with chained label confusions (A→B→C where the model
also confuses B with C), the frontier misses C entirely. Surfacing the whole
cluster gives the model examples of all transitively-confused boundaries.
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
# Max cluster size to surface — avoids flooding context with one giant cluster
_MAX_CLUSTER_MEMBERS = 8

# CJK Unicode ranges: CJK Unified, Hiragana, Katakana, Fullwidth/Halfwidth
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


def _bfs_cluster(seed_labels: set[str],
                 adj: dict[str, set[str]],
                 max_size: int) -> set[str]:
    """BFS from any seed label over the undirected confusion graph.

    Returns the union of connected-component members reachable from any seed,
    capped at max_size by priority-queue ordering (seeds first, then by
    discovery order).
    """
    visited: set[str] = set()
    queue: deque[str] = deque()
    for lbl in seed_labels:
        if lbl in adj:
            visited.add(lbl)
            queue.append(lbl)
    while queue and len(visited) < max_size:
        node = queue.popleft()
        for neighbour in adj.get(node, []):
            if neighbour not in visited and len(visited) < max_size:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


class ClusterConfusionMemory(MemorySystem):
    """Confusion-cluster retrieval via transitive graph expansion.

    The frontier system only disambiguates direct one-hop confusions (A was
    predicted when B was correct). This system builds an undirected confusion
    graph (edges A-B whenever A↔B appear in either direction of the confusion
    matrix) and at retrieval time expands the top candidate labels to their
    full connected component. One representative example per cluster member is
    injected before similarity-ranked fill, surfacing the entire boundary
    neighbourhood rather than just the direct confusors.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        # confusion[predicted][actual] = count
        self.confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _scored(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _build_adj(self) -> dict[str, set[str]]:
        """Build undirected adjacency from confusion matrix (both directions)."""
        adj: dict[str, set[str]] = defaultdict(set)
        for pred, actuals in self.confusion.items():
            for actual in actuals:
                if pred != actual:
                    adj[pred].add(actual)
                    adj[actual].add(pred)
        return adj

    def _cluster_for(self, labels: list[str]) -> set[str]:
        """Return the confusion-graph cluster containing any of the given labels."""
        adj = self._build_adj()
        seeds = set(labels) & set(adj.keys())
        if not seeds:
            return set()
        return _bfs_cluster(seeds, adj, _MAX_CLUSTER_MEMBERS)

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        ranked = self._scored(query)
        top_labels = [ex["target"] for _, _, ex in ranked[:_TOP_CANDIDATE_LABELS]]

        # Expand top candidate labels to their full confusion cluster
        cluster = self._cluster_for(top_labels)
        # Exclude the top candidate labels themselves — they are already well
        # represented by the similarity-ranked fill; we want the *other* cluster
        # members that the frontier system would miss
        extra_labels = cluster - set(top_labels)

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

        # Phase 1: one best-matching example per extra cluster member
        if extra_labels:
            per_label: dict[str, tuple[float, int, dict] | None] = {lbl: None for lbl in extra_labels}
            for score, idx, ex in ranked:
                lbl = ex["target"]
                if lbl in per_label and per_label[lbl] is None:
                    per_label[lbl] = (score, idx, ex)
                if all(v is not None for v in per_label.values()):
                    break
            # Inject cluster representatives ordered by similarity
            cluster_reps = sorted(
                (v for v in per_label.values() if v is not None),
                key=lambda t: (-t[0], t[1])
            )
            for score, idx, ex in cluster_reps:
                try_add(idx, ex)

        # Phase 2: similarity-ranked fill for remaining budget
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
                pred = r.get("prediction", "")
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
