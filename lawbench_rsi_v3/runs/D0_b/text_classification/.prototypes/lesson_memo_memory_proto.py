"""Prototype: LessonMemoMemory — LLM-synthesized lesson memos from error batches.

Mechanism under test:
- After accumulating errors, call LLM once to produce a compact lesson (<=5 bullet rules)
  that generalises across those errors.
- Store lessons in a list; at predict time retrieve top-k most relevant lessons by
  bigram Jaccard similarity and prepend them before examples.
- Variant A: store up to 20 lessons (one per error batch trigger)
- Variant B: merge/replace lessons when new errors are similar to past lesson source

Fake LLM: returns a canned lesson string for synthesis calls, and a canned answer for predict calls.
"""

import json
import re
from collections import defaultdict
from typing import Any


# ── same tokenizer as the frontier ──────────────────────────────────────────

def _tokenize(text: str) -> frozenset:
    tokens: set = set()
    tokens.update(re.findall(r"[A-Za-z0-9]+", text.lower()))
    cjk = re.findall(r"[一-鿿]", text)
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i] + cjk[i + 1])
    return frozenset(tokens)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── fake LLM ─────────────────────────────────────────────────────────────────

LESSON_RESPONSE = """{"lesson": "- When the fact mentions '单位' (company/organization), check whether the subject is an entity, not an individual — use unit-form charges like 单位行贿 instead of 行贿.\\n- Do NOT append '罪' to the charge name; the canonical form has no suffix.\\n- When someone buys stolen goods or repaints them to conceal origin, add 掩饰、隐瞒犯罪所得 alongside the primary charge.\\n- '非国家工作人员' means the person is not a state employee — use 非国家工作人员受贿, not 受贿.\\n- Electrical-cable theft that endangers public infrastructure is 破坏电力设备, not 盗窃."}"""

PREDICT_RESPONSE = '{"reasoning": "test", "final_answer": "诈骗"}'

LESSON_KEYWORD = "生成规律"  # trigger word in lesson-synthesis prompt


def fake_llm(prompt: str) -> str:
    if LESSON_KEYWORD in prompt:
        return LESSON_RESPONSE
    return PREDICT_RESPONSE


# ── lesson memo system (simplified) ──────────────────────────────────────────

_MAX_ERRORS_TO_SYNTHESIZE = 6   # trigger synthesis when this many new errors accumulate
_MAX_LESSONS = 20
_MAX_LESSON_INJECT = 3
_MAX_EXAMPLES = 15
MAX_CHARS = 30000

LESSON_PROMPT = """以下是模型在若干案例中的预测错误。请总结出最多5条简明的判案规律，帮助未来避免类似错误。
每条规律必须基于案件事实中可观测的信号，而不是简单重复错误信息。
用中文回答，格式如下：

{{"lesson": "- 规律1\\n- 规律2\\n- ..."}}

错误案例：
{error_cases}

请生成规律："""

PREDICT_PROMPT = """根据以下经验规律和示例，解决问题。

{lessons_section}{examples_section}

**问题：**
{input}

**说明：**
- 遵循示例中的模式
- 特别注意上方的规律，避免常见错误
- 以JSON格式回答

{{"reasoning": "[你的推理]", "final_answer": "[你的答案]"}}"""


class LessonMemoMemoryProto:
    def __init__(self, llm):
        self._llm = llm
        self.examples: list[dict] = []
        self.lessons: list[dict] = []   # {text, tokens, source_preview}
        self._pending_errors: list[dict] = []

    def _maybe_synthesize(self):
        if len(self._pending_errors) < _MAX_ERRORS_TO_SYNTHESIZE:
            return
        # Build error case text
        cases = []
        for e in self._pending_errors[:_MAX_ERRORS_TO_SYNTHESIZE]:
            preview = e["input"][:120].replace("\n", " ").replace("\r", "")
            cases.append(f"事实片段: {preview}\n预测: {e['prediction']}\n正确答案: {e['ground_truth']}")
        error_text = "\n---\n".join(cases)
        prompt = LESSON_PROMPT.format(error_cases=error_text)
        resp = self._llm(prompt)
        try:
            lesson_text = json.loads(resp).get("lesson", "")
        except Exception:
            m = re.search(r'"lesson"\s*:\s*"(.*?)"', resp, re.DOTALL)
            lesson_text = m.group(1) if m else ""
        if lesson_text:
            # tokens from source errors for relevance matching
            combined_input = " ".join(e["input"][:200] for e in self._pending_errors[:_MAX_ERRORS_TO_SYNTHESIZE])
            self.lessons.append({
                "text": lesson_text,
                "tokens": _tokenize(combined_input),
                "source_preview": self._pending_errors[0]["input"][:40],
            })
            if len(self.lessons) > _MAX_LESSONS:
                self.lessons.pop(0)  # evict oldest
        self._pending_errors = self._pending_errors[_MAX_ERRORS_TO_SYNTHESIZE:]

    def learn_from_batch(self, batch_results: list[dict]) -> None:
        for r in batch_results:
            self.examples.append({
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": _tokenize(r["input"]),
            })
            if not r.get("was_correct", True):
                self._pending_errors.append(r)
        self._maybe_synthesize()

    def _top_lessons(self, q_tok: frozenset) -> list[str]:
        if not self.lessons:
            return []
        scored = [(l, _jaccard(q_tok, l["tokens"])) for l in self.lessons]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [l["text"] for l, _ in scored[:_MAX_LESSON_INJECT]]

    def _top_examples(self, q_tok: frozenset) -> list[str]:
        scored = [(_jaccard(q_tok, ex["tokens"]), i, ex) for i, ex in enumerate(self.examples)]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        parts, total = [], 0
        for _, _, ex in scored:
            part = f"Q: {ex['input'][:300]}\nA: {ex['target']}"
            if total + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total += len(part) + 2
            if len(parts) >= _MAX_EXAMPLES:
                break
        return parts

    def predict(self, input_text: str) -> str:
        q_tok = _tokenize(input_text)
        lessons = self._top_lessons(q_tok)
        examples = self._top_examples(q_tok)

        lessons_section = ""
        if lessons:
            bullet = "\n".join(lessons)
            lessons_section = f"**经验规律（请特别注意）：**\n{bullet}\n\n"

        examples_section = ""
        if examples:
            examples_section = "**示例：**\n" + "\n\n".join(examples) + "\n\n"

        prompt = PREDICT_PROMPT.format(
            lessons_section=lessons_section,
            examples_section=examples_section,
            input=input_text,
        )
        resp = self._llm(prompt)
        try:
            return json.loads(resp).get("final_answer", "")
        except Exception:
            m = re.search(r'"final_answer"\s*:\s*"([^"]*)"', resp)
            return m.group(1) if m else ""


# ── tests ─────────────────────────────────────────────────────────────────────

def run_tests():
    # Real error examples from score diagnostics (bigram_confusion_memory)
    errors = [
        {"input": "被告单位通辽市某商业广场有限公司及其法定代表人沙某某，利用另案犯罪嫌疑人张某某的职务便利，以通辽市政府办公厅的名义出具了推荐函",
         "prediction": "行贿;非法采矿;伪造国家机关公文、证件、印章",
         "ground_truth": "单位行贿",
         "was_correct": False},
        {"input": "被告人林某为办理该处房产的土地证，将房产证从马某某手中借回，伪造了假的房产证交还给马某某",
         "prediction": "伪造国家机关证件罪;诈骗罪",
         "ground_truth": "伪造、变造、买卖国家机关公文、证件、印章",
         "was_correct": False},
        {"input": "被告人韩某某在担任西宁某物业管理有限公司副总经理期间，在某小区保障性住房项目改造过程中利用职务之便利，非法收受闫某的行贿款",
         "prediction": "受贿",
         "ground_truth": "非国家工作人员受贿",
         "was_correct": False},
        {"input": "李某甲以牟利为目的，盗割正在使用中的公共照明电线，危害公共安全",
         "prediction": "破坏公共设施",
         "ground_truth": "破坏电力设备",
         "was_correct": False},
        {"input": "被告人何某在肇庆市矶西路江南副产品批发市场门口，偷走了被害人罗某斌一辆红色大运牌三轮摩托车，购买了一辆黑色雅马哈牌摩托车将该车车架及轮毂喷成黄色",
         "prediction": "盗窃",
         "ground_truth": "盗窃;掩饰、隐瞒犯罪所得、犯罪所得收益",
         "was_correct": False},
        {"input": "被告人郭某受雇于同案人，在本市白云区均禾街富力城销售假冒GIVENCHY、LOUISVUITTON、HERMES等注册商标的皮鞋",
         "prediction": "侵犯注册商标专用权",
         "ground_truth": "销售假冒注册商标的商品",
         "was_correct": False},
    ]

    correct = [
        {"input": "被告人叶某甲从青田县季宅乡黄放口村溪滩边摘取几个罂粟果，后将上述罂粟果种植在青田县季宅乡华坦村后山自家农田中",
         "prediction": "非法种植毒品原植物",
         "ground_truth": "非法种植毒品原植物",
         "was_correct": True},
        {"input": "被告人何某某在三台县建设镇干坝王村六组蒋家岩坡，私自焚烧秸秆不慎引发山火",
         "prediction": "失火",
         "ground_truth": "失火",
         "was_correct": True},
    ]

    sys = LessonMemoMemoryProto(fake_llm)

    # --- Test 1: cold start (no examples, no lessons) ---
    pred = sys.predict("被告人张某某谎称以租车的名义将被害人李某某的轿车骗走后，张某某将骗得的轿车抵押给朋友借款")
    print(f"[cold start] prediction: '{pred}'")
    assert pred == "诈骗", f"expected 诈骗, got {pred}"

    # --- Test 2: learn from batch (including 6 errors → synthesis triggered) ---
    sys.learn_from_batch(errors + correct)
    print(f"[after batch] lessons count: {len(sys.lessons)}")
    print(f"[after batch] examples count: {len(sys.examples)}")
    assert len(sys.lessons) >= 1, "expected at least 1 lesson after 6 errors"
    assert len(sys.examples) == len(errors) + len(correct)

    # --- Test 3: predict with lessons injected ---
    test_input = "被告单位某公司及其法定代表人沙某某，以政府办公厅的名义出具了推荐函，使不具备资质的公司得以从事石油开采，并向官员支付了不正当报酬"
    pred2 = sys.predict(test_input)
    print(f"[with lessons] prediction: '{pred2}'")
    assert pred2 == "诈骗"  # fake LLM always returns 诈骗 for non-lesson calls

    # --- Test 4: lesson content quality check ---
    lesson_text = sys.lessons[0]["text"]
    print(f"[lesson text preview]: {lesson_text[:100]}")
    assert len(lesson_text) > 10, "lesson should have real content"

    # --- Test 5: lesson retrieval by similarity ---
    # A query about 单位/company context should retrieve the lesson
    q_tok = _tokenize("被告单位某商业广场有限公司法定代表人，以公司名义支付报酬")
    top = sys._top_lessons(q_tok)
    print(f"[top lessons for unit-bribery query]: {len(top)} lessons")

    # --- Variant B: check that lessons don't grow unbounded ---
    for _ in range(5):
        sys.learn_from_batch(errors)  # each triggers synthesis
    print(f"[after 5 more batches] lessons count: {len(sys.lessons)} (max {_MAX_LESSONS})")
    assert len(sys.lessons) <= _MAX_LESSONS

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
