"""MMR Memory — Maximum Marginal Relevance retrieval.

At each greedy step, selects the stored example that maximises:

    score(e) = λ * sim(e, query) − (1−λ) * max_{s ∈ selected} sim(e, s)

This balances relevance to the current query against redundancy with
already-chosen examples, so the context window receives a diverse,
topically-relevant set without requiring explicit label bookkeeping.

Unlike pure similarity ranking (which clusters on the most frequent
topic) or round-robin (which treats every label as equally useful),
MMR adapts the diversity penalty to the actual redundancy observed in
the retrieved set.

Similarity is measured on raw_question only to avoid dilution from
shared instruction boilerplate. λ=0.6 weights relevance slightly
over diversity, preserving topical focus while penalising near-duplicate
examples.

Reference: Carbonell & Goldstein, SIGIR 1998.
"""

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
LAMBDA = 0.6  # relevance weight; 1-LAMBDA is diversity weight


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class MmrMemory(MemorySystem):
    """Greedy MMR selection: relevant to query, diverse from already-chosen examples."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []

    def _select_mmr(self, query: str) -> list[str]:
        if not self.examples:
            return []

        q_tok = _tokenize(query)

        # Pre-compute relevance and token sets for all stored examples
        pool = [
            {
                "ex": ex,
                "tok": ex["tokens"],
                "rel": _jaccard(q_tok, ex["tokens"]),
                "idx": i,
            }
            for i, ex in enumerate(self.examples)
        ]

        selected_parts: list[str] = []
        selected_toks: list[frozenset[str]] = []
        total_chars = 0
        remaining = list(range(len(pool)))

        while remaining:
            best_score: float | None = None
            best_ri: int | None = None

            for ri in remaining:
                item = pool[ri]
                rel = item["rel"]
                if selected_toks:
                    redundancy = max(_jaccard(item["tok"], st) for st in selected_toks)
                else:
                    redundancy = 0.0
                mmr = LAMBDA * rel - (1 - LAMBDA) * redundancy

                if best_score is None or mmr > best_score:
                    best_score = mmr
                    best_ri = ri

            item = pool[best_ri]
            ex = item["ex"]
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"

            if total_chars + len(part) + 2 > MAX_CHARS:
                break

            selected_parts.append(part)
            selected_toks.append(item["tok"])
            total_chars += len(part) + 2
            remaining.remove(best_ri)

        return selected_parts

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = self._select_mmr(input)
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
        return sum(len(p) + 2 for p in self._select_mmr(""))

    def get_state(self) -> str:
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
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                raw_q = restored.get("raw_question", restored.get("input", ""))
                restored["tokens"] = _tokenize(raw_q)
            self.examples.append(restored)
