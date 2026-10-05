"""Similarity Retrieval Memory - retrieve examples most similar to the current query.

At predict time, ranks all stored examples by token-level Jaccard similarity
to the current query, then fills the context budget with the highest-scoring
examples. This targets tasks with structured inputs (e.g. domain-specific
syntax or vocabulary) where the most similar training examples are more
informative than the most recent ones.

Selection algorithm:
  - Tokenise query and every stored example's raw_question by alphanumeric
    sequences (case-folded).
  - Score each example by Jaccard(query_tokens, example_tokens).
  - Sort descending by score (break ties by recency).
  - Fill context greedily until char budget is exhausted.

This is a fundamentally different retrieval mechanism from fewshot_all (recency),
contrastive_error_memory (recency + error priority), and reflexion_memory (LLM
synthesis). No LLM is called in learn_from_batch, keeping learning cost zero.
"""

import hashlib
import json
import re
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


def _tokenize(text: str) -> frozenset[str]:
    """Return lowercase alphanumeric tokens from text."""
    return frozenset(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


class SimilarityRetrievalMemory(MemorySystem):
    """Retrieve examples by Jaccard token similarity to the current query."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        # Each entry: {"input": ..., "target": ..., "raw_question": ..., "tokens": frozenset}
        self.examples: list[dict[str, Any]] = []

    def _select_by_similarity(self, query: str) -> list[str]:
        """Rank examples by similarity to query, fill context budget greedily."""
        if not self.examples:
            return []

        q_tokens = _tokenize(query)

        # Score and rank: (score, arrival_index, example)
        # Use arrival_index as secondary sort key so ties go to most recent
        scored = [
            (_jaccard(q_tokens, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        parts: list[str] = []
        total_chars = 0
        for _score, _idx, ex in scored:
            question = ex.get("raw_question", ex["input"])
            part = f"Q: {question}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > MAX_CHARS:
                break
            parts.append(part)
            total_chars += len(part) + 2

        return parts

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = self._select_by_similarity(input)
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
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            ex: dict[str, Any] = {
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": _tokenize(raw_q),
            }
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

    def get_context_length(self) -> int:
        # Use a short generic query for the length estimate
        return sum(len(p) + 2 for p in self._select_by_similarity(""))

    def get_state(self) -> str:
        # frozenset is not JSON-serialisable — convert to sorted list for storage
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        return json.dumps({"examples": serialisable}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        for ex in data.get("examples", []):
            restored = dict(ex)
            # Restore tokens as frozenset from stored list
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                raw_q = restored.get("raw_question", restored.get("input", ""))
                restored["tokens"] = _tokenize(raw_q)
            self.examples.append(restored)
