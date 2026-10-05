"""Contrastive Error Pair Memory — stores raw error pairs instead of LLM lessons.

Mechanism change from reflexion_lesson_memory:

ReflexionLessonMemory calls the LLM at learn time to abstract each error into a
rule. The abstraction costs an LLM call and may lose case-specific signal when
the distinguishing feature is subtle. ContrastiveErrorPairMemory instead stores
the raw wrong-prediction record directly: the predicted label, the correct label,
and a case excerpt. At predict time the system retrieves the most similar past
mistakes by Jaccard similarity and surfaces them in a dedicated section before
the examples, so the model sees a structurally distinct "WRONG / CORRECT" signal
rather than relying on it to infer the correction from a surface-similar positive
example.

Axes changed:
- A (Prompt template): new "Common mistakes — AVOID these errors" section with
  explicit WRONG→CORRECT formatting, replacing the lesson bullet format
- B (Memory content): raw error pairs stored (no LLM-generated text)
- F (LLM usage in learning): removed — no LLM call during learn_from_batch

Cold start: works fine; error section is simply empty until first error is seen.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PREDICT_TEMPLATE = """Solve the classification problem below.

{contrastive_section}{examples_section}**Problem:**
{input}

**Instructions:**
- If a mistake above matches this case pattern, use the CORRECT label shown
- A case may require MORE THAN ONE charge; output all that apply, semicolon-separated
- Use the EXACT official label — do not append or drop characters
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[exact label(s), semicolon-separated if multiple]"}}"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CHARS = 30000
# Max error pairs kept (most recent first, oldest dropped)
MAX_ERROR_PAIRS = 40
# Character budget for the contrastive section
CONTRASTIVE_BUDGET = 10000
# Case excerpt length stored per error pair
_EXCERPT_LEN = 300


def _tokenize(text: str) -> frozenset[str]:
    """CJK character bigrams + ASCII tokens for fallback similarity."""
    chars = re.findall(r"[一-鿿㐀-䶿豈-﫿]", text)
    if len(chars) >= 2:
        return frozenset(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return frozenset(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Memory system
# ---------------------------------------------------------------------------

class ContrastiveErrorPairMemory(MemorySystem):
    """Stores raw wrong-prediction records; retrieves similar ones at predict time."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        # Error pairs: {predicted, actual, excerpt, tokens}
        self.error_pairs: list[dict[str, Any]] = []
        # All examples for similarity-based fallback fill
        self.examples: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Learning: no LLM call — store error records directly
    # ------------------------------------------------------------------

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            tok = _tokenize(raw_q)
            self.examples.append({
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": tok,
                "raw_question": raw_q,
            })

            if not r.get("was_correct", True):
                pred = r.get("prediction", "")
                gt = r["ground_truth"]
                if pred and pred != gt:
                    self.error_pairs.append({
                        "predicted": pred,
                        "actual": gt,
                        "excerpt": r["input"][:_EXCERPT_LEN],
                        "tokens": tok,
                    })
                    if len(self.error_pairs) > MAX_ERROR_PAIRS:
                        self.error_pairs = self.error_pairs[-MAX_ERROR_PAIRS:]

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _build_contrastive_section(self, query: str) -> tuple[str, int]:
        """Top-k error pairs by similarity; returns (section_text, chars_used)."""
        if not self.error_pairs:
            return "", 0
        q_tok = _tokenize(query)
        ranked = sorted(
            [((_jaccard(q_tok, ep["tokens"]), i), ep)
             for i, ep in enumerate(self.error_pairs)],
            key=lambda x: x[0],
            reverse=True,
        )
        lines = ["**Common mistakes on similar cases — AVOID these errors:**"]
        total = len(lines[0])
        used = 0
        for _, ep in ranked[:8]:
            line = (
                f"- WRONG: {ep['predicted']} → CORRECT: {ep['actual']}\n"
                f"  Case excerpt: {ep['excerpt']}"
            )
            if total + len(line) + 2 > CONTRASTIVE_BUDGET:
                break
            lines.append(line)
            total += len(line) + 2
            used += 1
        if used == 0:
            return "", 0
        return "\n".join(lines) + "\n\n", total

    def _build_examples_section(self, query: str, budget: int) -> str:
        """Fill char budget with similarity-ranked raw examples."""
        if not self.examples:
            return ""
        q_tok = _tokenize(query)
        ranked = sorted(
            [(_jaccard(q_tok, ex["tokens"]), i, ex) for i, ex in enumerate(self.examples)],
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )
        parts: list[str] = []
        total = 0
        for _, _, ex in ranked:
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total + len(part) + 2 > budget:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts) + "\n\n" if parts else ""

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        contrastive_sec, contrast_chars = self._build_contrastive_section(input)
        examples_budget = max(0, MAX_CHARS - contrast_chars - 500)
        examples_sec = self._build_examples_section(input, examples_budget)

        prompt = PREDICT_TEMPLATE.format(
            contrastive_section=contrastive_sec,
            examples_section=examples_sec,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_error_pairs": len(self.error_pairs),
            "num_examples": len(self.examples),
            "contrastive_chars": contrast_chars,
        }

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def get_state(self) -> str:
        def _ser(item: dict) -> dict:
            return {k: (sorted(v) if isinstance(v, frozenset) else v)
                    for k, v in item.items()}
        return json.dumps({
            "error_pairs": [_ser(ep) for ep in self.error_pairs],
            "examples": [_ser(ex) for ex in self.examples],
        }, indent=2, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.error_pairs = []
        for ep in data.get("error_pairs", []):
            restored = dict(ep)
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                restored["tokens"] = _tokenize(restored.get("excerpt", ""))
            self.error_pairs.append(restored)
        self.examples = []
        for ex in data.get("examples", []):
            restored = dict(ex)
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                raw_q = restored.get("raw_question", restored.get("input", ""))
                restored["tokens"] = _tokenize(raw_q)
            self.examples.append(restored)
