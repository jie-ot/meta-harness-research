"""Reflexion Memory - LLM-distilled lessons from error batches.

Inspired by Reflexion (Shinn et al., 2023). After each training batch,
calls the LLM to distill recurring error patterns into compact, actionable
lessons. These lessons are injected at the top of the prompt, giving the
model explicit meta-guidance rather than raw examples alone.

Mechanism (different from all fewshot baselines):
  learn_from_batch: on every batch that has errors, call the LLM once to
    synthesize the errors into <=5 lessons. Deduplicate against existing
    lessons by semantic overlap detection (LLM judge). Cap at 15 total lessons.
  predict: inject lessons section + a small set of recent examples.
"""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

LESSON_SYNTHESIS_PROMPT = """You are analyzing classification mistakes to extract reusable patterns.

Here are recent errors (wrong prediction → correct answer):
{error_pairs}

Write up to 5 concise, general lessons that would help avoid these mistakes in the future.
Each lesson should be a single sentence describing a pattern, signal, or rule.
Do NOT mention specific texts from above — generalize the pattern.
Respond as a JSON array of strings:
{{"lessons": ["lesson 1", "lesson 2", ...]}}"""

PREDICT_PROMPT = """Solve the classification problem below.

{lessons_section}{examples_section}
**Problem:**
{input}

Respond in JSON format:
{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_LESSONS = 15
MAX_EXAMPLES = 8
MAX_EXAMPLE_CHARS = 8000


class ReflexionMemory(MemorySystem):
    """LLM-distilled lesson memory inspired by Reflexion."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.lessons: list[str] = []
        self.examples: list[dict[str, str]] = []

    def _format_lessons_section(self) -> str:
        if not self.lessons:
            return ""
        lines = ["## Lessons from past mistakes:"]
        for i, lesson in enumerate(self.lessons, 1):
            lines.append(f"{i}. {lesson}")
        return "\n".join(lines) + "\n\n"

    def _format_examples_section(self) -> str:
        if not self.examples:
            return ""
        parts = []
        total = 0
        for ex in self.examples[-MAX_EXAMPLES:]:
            question = ex.get("raw_question", ex["input"])
            part = f"Q: {question}\nA: {ex['target']}"
            if total + len(part) > MAX_EXAMPLE_CHARS:
                break
            parts.append(part)
            total += len(part) + 2
        if not parts:
            return ""
        return "## Examples:\n" + "\n\n".join(parts) + "\n\n"

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        lessons_section = self._format_lessons_section()
        examples_section = self._format_examples_section()
        prompt = PREDICT_PROMPT.format(
            lessons_section=lessons_section,
            examples_section=examples_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_lessons": len(self.lessons),
            "num_examples": len(self.examples),
        }

    def _synthesize_lessons(self, errors: list[dict[str, Any]]) -> list[str]:
        """Call LLM to distill error patterns into lessons."""
        if not errors:
            return []

        # Format error pairs for the synthesis prompt
        pairs = []
        for e in errors[-10:]:  # use at most 10 recent errors
            question = e.get("raw_question", e["input"])
            # Truncate long inputs to keep synthesis prompt manageable
            if len(question) > 300:
                question = question[:300] + "..."
            pairs.append(
                f"  Input: {question}\n"
                f"  Predicted: {e['prediction']}\n"
                f"  Correct: {e['ground_truth']}"
            )

        error_pairs_str = "\n\n".join(pairs)
        prompt = LESSON_SYNTHESIS_PROMPT.format(error_pairs=error_pairs_str)

        try:
            response = self.call_llm(prompt)
            # Extract as JSON array field
            raw = extract_json_field(response, "lessons")
            # The field itself may be a JSON array string or already a list
            if raw.startswith("["):
                new_lessons = json.loads(raw)
            else:
                # Fallback: treat as single lesson
                new_lessons = [raw] if raw else []
            return [str(l).strip() for l in new_lessons if str(l).strip()]
        except Exception:
            return []

    def _deduplicate_lessons(self, new_lessons: list[str]) -> list[str]:
        """Simple dedup: skip new lessons that share >4 words with existing ones."""
        unique = []
        existing_words = set()
        for lesson in self.lessons:
            existing_words.update(lesson.lower().split())

        for new in new_lessons:
            new_words = set(new.lower().split())
            overlap = len(new_words & existing_words)
            # Skip if more than half the words already appear in existing lessons
            if len(new_words) > 0 and overlap / len(new_words) < 0.6:
                unique.append(new)
                existing_words.update(new_words)

        return unique

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        """Accumulate examples and synthesize lessons from errors."""
        errors = [r for r in batch_results if not r["was_correct"]]

        # Always store raw examples
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

        # Only synthesize lessons if there are errors in this batch
        if errors:
            new_lessons = self._synthesize_lessons(errors)
            deduped = self._deduplicate_lessons(new_lessons)
            self.lessons.extend(deduped)
            # Cap total lesson count
            if len(self.lessons) > MAX_LESSONS:
                self.lessons = self.lessons[-MAX_LESSONS:]

    def get_context_length(self) -> int:
        return len(self._format_lessons_section()) + len(self._format_examples_section())

    def get_state(self) -> str:
        return json.dumps({"lessons": self.lessons, "examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.lessons = data.get("lessons", [])
        self.examples = data.get("examples", [])
