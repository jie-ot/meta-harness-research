"""LessonMemoMemory — LLM-synthesized lesson memos from error batches.

Axis F exploitation on top of bigram_confusion_memory: the LLM is called
once per error batch (after accumulating _MAX_ERRORS_TO_SYNTHESIZE wrong
predictions) to produce a compact set of generalizable rules. Rules are
stored with bigram tokens derived from their source errors. At predict time
the top-3 most relevant rules are injected as a "Key rules" section before
the examples, steering the model toward correct canonical label formats and
away from the fine-grained boundary errors that dominate remaining failures.

Unlike raw contrastive pairs, synthesized rules generalize: a rule that
identifies "when the subject is a legal entity, use the unit-form charge"
applies to any unseen entity-bribery case, not just the specific example
that triggered it.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

MAX_CHARS = 30000
_MAX_ERRORS_TO_SYNTHESIZE = 6   # trigger synthesis after this many new errors
_MAX_LESSONS = 20               # rolling cap on stored lessons (evict oldest)
_MAX_LESSON_INJECT = 3          # top-k lessons to inject per prediction
_MAX_EXAMPLES = 15              # similarity-ranked examples to append after lessons


def _tokenize(text: str) -> frozenset[str]:
    """Chinese character bigrams + ASCII words."""
    tokens: set[str] = set()
    tokens.update(re.findall(r"[A-Za-z0-9]+", text.lower()))
    cjk_chars = re.findall(r"[一-鿿]", text)
    for i in range(len(cjk_chars) - 1):
        tokens.add(cjk_chars[i] + cjk_chars[i + 1])
    return frozenset(tokens)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


LESSON_PROMPT = """以下是模型在若干案例中的预测错误，请总结出最多5条简明的判案规律，帮助未来避免类似错误。
每条规律必须基于案件事实中可观测的信号，而非简单重复错误本身。
用中文回答，格式如下：

{{"lesson": "- 规律1\\n- 规律2\\n- ..."}}

错误案例：
{error_cases}

请生成规律："""

PREDICT_PROMPT = """Solve the problem below based on the rules and examples provided.

{lessons_section}{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples
- Pay special attention to the key rules above to avoid common mistakes
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""


class LessonMemoMemory(MemorySystem):
    """LLM-synthesized lesson memos injected at predict time.

    learn_from_batch accumulates wrong predictions; once _MAX_ERRORS_TO_SYNTHESIZE
    have collected, a single LLM call produces a compact set of generalizable rules
    (a "lesson"). Lessons are stored with bigram tokens from their source errors so
    the most contextually relevant ones can be retrieved at predict time and
    prepended before the similarity-ranked examples.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.lessons: list[dict[str, Any]] = []      # {text, tokens, source_preview}
        self._pending_errors: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lesson synthesis
    # ------------------------------------------------------------------

    def _maybe_synthesize(self) -> None:
        if len(self._pending_errors) < _MAX_ERRORS_TO_SYNTHESIZE:
            return
        batch = self._pending_errors[:_MAX_ERRORS_TO_SYNTHESIZE]
        cases = []
        for e in batch:
            preview = e["input"][:120].replace("\n", " ").replace("\r", "")
            cases.append(
                f"事实片段: {preview}\n预测: {e.get('prediction', '')}\n正确答案: {e['ground_truth']}"
            )
        prompt = LESSON_PROMPT.format(error_cases="\n---\n".join(cases))
        resp = self.call_llm(prompt)
        lesson_text = extract_json_field(resp, "lesson")
        if lesson_text:
            combined = " ".join(e["input"][:200] for e in batch)
            self.lessons.append({
                "text": lesson_text,
                "tokens": _tokenize(combined),
                "source_preview": batch[0]["input"][:40],
            })
            if len(self.lessons) > _MAX_LESSONS:
                self.lessons.pop(0)
        self._pending_errors = self._pending_errors[_MAX_ERRORS_TO_SYNTHESIZE:]

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _top_lessons(self, q_tok: frozenset[str]) -> list[str]:
        if not self.lessons:
            return []
        scored = sorted(
            self.lessons,
            key=lambda l: _jaccard(q_tok, l["tokens"]),
            reverse=True,
        )
        return [l["text"] for l in scored[:_MAX_LESSON_INJECT]]

    def _top_examples(self, q_tok: frozenset[str]) -> list[str]:
        scored = sorted(
            enumerate(self.examples),
            key=lambda ie: (_jaccard(q_tok, ie[1]["tokens"]), ie[0]),
            reverse=True,
        )
        parts: list[str] = []
        total = 0
        for _, ex in scored:
            part = f"Q: {ex['input'][:300]}\nA: {ex['target']}"
            if total + len(part) + 2 > MAX_CHARS:
                break
            parts.append(part)
            total += len(part) + 2
            if len(parts) >= _MAX_EXAMPLES:
                break
        return parts

    # ------------------------------------------------------------------
    # MemorySystem interface
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        q_tok = _tokenize(input)
        lessons = self._top_lessons(q_tok)
        examples = self._top_examples(q_tok)

        lessons_section = ""
        if lessons:
            lessons_section = "**Key rules (pay special attention):**\n" + "\n".join(lessons) + "\n\n"

        examples_section = ""
        if examples:
            examples_section = "**Examples:**\n" + "\n\n".join(examples) + "\n\n"

        prompt = PREDICT_PROMPT.format(
            lessons_section=lessons_section,
            examples_section=examples_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_lessons": len(self.lessons),
            "num_lessons_injected": len(lessons),
            "num_examples_injected": len(examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            self.examples.append({
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": _tokenize(r.get("raw_question", r["input"])),
            })
            if not r.get("was_correct", True):
                self._pending_errors.append(r)
        self._maybe_synthesize()

    def get_state(self) -> str:
        return json.dumps({
            "examples": [
                {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
                for ex in self.examples
            ],
            "lessons": [
                {
                    "text": l["text"],
                    "tokens": sorted(l["tokens"]),
                    "source_preview": l["source_preview"],
                }
                for l in self.lessons
            ],
            "pending_errors": self._pending_errors,
        }, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        for ex in data.get("examples", []):
            restored = dict(ex)
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                restored["tokens"] = _tokenize(restored.get("input", ""))
            self.examples.append(restored)

        self.lessons = []
        for l in data.get("lessons", []):
            self.lessons.append({
                "text": l["text"],
                "tokens": frozenset(l.get("tokens", [])),
                "source_preview": l.get("source_preview", ""),
            })

        self._pending_errors = data.get("pending_errors", [])
