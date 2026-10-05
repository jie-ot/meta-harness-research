"""Anchor-Last Memory.

Builds on adaptive_tokenizer_confusion_memory but changes the prompt
architecture: the last K slots in the context window are reserved for
"anchor" examples — one highest-similarity example per top candidate
label — placed immediately before the question.  The rest of the context
budget is filled with similarity-ranked examples from the top.

Hypothesis: transformer attention has a recency / proximity bias, so
examples that appear just before the question exert more influence on
the prediction than examples buried at the top of the context.  By
reserving the final positions for the best representative of each likely
label we amplify the discrimination signal precisely where the model
attends most strongly.

Retrieval algorithm:
1. Rank all stored examples by adaptive Jaccard against the query.
2. Identify top-K candidate labels from the ranked list.
3. Pick the single highest-similarity example per candidate label as an
   "anchor"; reserve these for the final positions.
4. Fill the budget *before* the anchors with similarity-ranked examples
   (anchors excluded from fill to avoid duplication).
5. Concatenate: [fill examples ... anchor examples] → prompt.
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
# How many candidate labels to anchor; covers most tasks without over-reserving slots
_NUM_ANCHORS = 4

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


class AnchorLastMemory(MemorySystem):
    """Proximity-biased retrieval: anchor examples placed last in context."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []

    def _ranked(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        ranked = self._ranked(query)

        # --- Step 1: identify anchor examples (best per top-N candidate labels) ---
        seen_labels: list[str] = []
        anchors: dict[str, tuple[int, dict]] = {}  # label -> (idx, ex)
        for score, idx, ex in ranked:
            lbl = ex["target"]
            if lbl not in anchors:
                anchors[lbl] = (idx, ex)
                seen_labels.append(lbl)
            if len(anchors) == _NUM_ANCHORS:
                break

        anchor_indices: set[int] = {idx for (idx, _) in anchors.values()}

        # Build anchor parts (ordered by candidate label appearance in ranked list)
        anchor_parts: list[str] = []
        anchor_chars = 0
        for lbl in seen_labels:
            if lbl in anchors:
                _, ex = anchors[lbl]
                q = ex.get("raw_question", ex["input"])
                p = f"Q: {q}\nA: {ex['target']}"
                anchor_parts.append(p)
                anchor_chars += len(p) + 2

        # --- Step 2: fill remaining budget before anchors ---
        fill_budget = MAX_CHARS - anchor_chars
        fill_parts: list[str] = []
        fill_total = 0
        for _, idx, ex in ranked:
            if fill_total >= fill_budget:
                break
            if idx in anchor_indices:
                continue
            q = ex.get("raw_question", ex["input"])
            p = f"Q: {q}\nA: {ex['target']}"
            if fill_total + len(p) + 2 > fill_budget:
                break
            fill_parts.append(p)
            fill_total += len(p) + 2

        # Anchors go LAST (closest to the question)
        return fill_parts + anchor_parts

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
                restored["tokens"] = _tokenize_adaptive(raw_q)
            self.examples.append(restored)
