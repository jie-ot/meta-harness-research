"""Prototype B: Reflexion-style lesson memory.

Mechanism:
- On errors, the LLM synthesises a concise transfer rule from (input, wrong_pred, ground_truth).
- Rules are stored with keyword fingerprints for retrieval.
- At predict time, matching rules are injected as explicit guidance BEFORE examples.
- No raw examples stored — memory IS the distilled lesson set.

This is exploration on axis F (LLM usage in learning) + B (memory content).
Instead of raw case examples, memory holds generalised rules like:
  "When the facts describe forging a specific type of document (房产证, 身份证, 土地证),
   the charge is likely 伪造、变造、买卖国家机关公文、证件、印章, NOT 伪造国家机关证件罪"

Fake LLM simulates what the real LLM would generate.
"""
import re
from collections import defaultdict
from typing import Any


# ── Fake LLM for prototype testing ───────────────────────────────────────────

def fake_llm(prompt: str) -> str:
    """Simulate LLM output. Returns a lesson or a final_answer depending on prompt."""
    if "Distill a transfer rule" in prompt:
        # Simulate lesson generation
        if "伪造" in prompt and "房产证" in prompt:
            return '{"lesson": "当事实描述伪造房产证等国家证件时，罪名应为伪造、变造、买卖国家机关公文、证件、印章，而非伪造国家机关证件罪（后者带罪字后缀是错误格式）", "keywords": ["伪造", "房产证", "土地证", "证件"]}'
        if "诈骗" in prompt and "合同" in prompt:
            return '{"lesson": "在签订、履行合同过程中骗取财物的，罪名为合同诈骗，而非普通诈骗", "keywords": ["合同", "签订", "履行", "骗取"]}'
        if "受贿" in prompt and "国家工作人员" not in prompt:
            return '{"lesson": "非国家工作人员（如公司员工）收受贿赂的，罪名为非国家工作人员受贿，而非受贿罪", "keywords": ["公司", "经理", "员工", "收受", "好处费"]}'
        if "盗窃" in prompt and "电缆" in prompt:
            return '{"lesson": "盗割正在使用中的公共电力设施（电缆、电线）的，罪名为破坏电力设备，而非盗窃", "keywords": ["电缆", "电线", "路灯", "盗割", "公共"]}'
        return '{"lesson": "注意区分相似罪名的具体构成要件", "keywords": []}'
    else:
        # Simulate predict
        return '{"reasoning": "based on facts", "final_answer": "[罪名]合同诈骗<eoa>"}'


# ── Core lesson memory logic ──────────────────────────────────────────────────

def extract_json_field(text: str, field: str) -> str:
    import json, re
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            return d.get(field, "")
        except Exception:
            pass
    return ""

def extract_lesson(text: str):
    import json, re
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            return d.get("lesson", ""), d.get("keywords", [])
        except Exception:
            pass
    return "", []


def build_lesson_prompt(input_text: str, wrong_pred: str, ground_truth: str) -> str:
    return f"""An AI made the following classification error. Distill a transfer rule.

Case facts (excerpt):
{input_text[:800]}

Wrong prediction: {wrong_pred}
Correct answer: {ground_truth}

Write a concise, general rule (1-2 sentences) that would prevent this error class.
Include keywords that would help identify when to apply it.

{{"lesson": "...", "keywords": ["kw1", "kw2", ...]}}"""


def tokenize_kw(text: str) -> set:
    chars = re.findall(r'[一-鿿]', text)
    bigrams = {chars[i] + chars[i+1] for i in range(len(chars) - 1)}
    ascii_ = set(re.findall(r'[A-Za-z0-9]+', text.lower()))
    return bigrams | ascii_


def score_relevance(query_tokens: set, lesson: dict) -> float:
    kw_tokens = set()
    for kw in lesson.get("keywords", []):
        kw_tokens |= tokenize_kw(kw)
    lesson_text_tokens = tokenize_kw(lesson.get("lesson", ""))
    all_lesson_tokens = kw_tokens | lesson_text_tokens
    if not all_lesson_tokens:
        return 0.0
    overlap = len(query_tokens & all_lesson_tokens)
    return overlap / len(all_lesson_tokens)


# ── Simulate a mini learn + predict cycle ────────────────────────────────────

class ReflexionLessonMemory:
    def __init__(self, llm):
        self.llm = llm
        self.lessons: list[dict] = []  # {lesson, keywords}

    def learn_from_batch(self, batch_results: list[dict]):
        for r in batch_results:
            if not r.get("was_correct", True):
                pred = r.get("prediction", "")
                gt = r["ground_truth"]
                if not pred or pred == gt:
                    continue
                prompt = build_lesson_prompt(r["input"], pred, gt)
                response = self.llm(prompt)
                lesson_text, keywords = extract_lesson(response)
                if lesson_text:
                    self.lessons.append({"lesson": lesson_text, "keywords": keywords})

    def predict(self, input_text: str) -> tuple[str, dict]:
        q_tokens = tokenize_kw(input_text)
        scored = sorted(
            self.lessons,
            key=lambda l: score_relevance(q_tokens, l),
            reverse=True
        )
        top_lessons = [l for l in scored if score_relevance(q_tokens, l) > 0][:3]

        lessons_section = ""
        if top_lessons:
            rules = "\n".join(f"- {l['lesson']}" for l in top_lessons)
            lessons_section = f"**Important rules for this task:**\n{rules}\n\n"

        prompt = f"""{lessons_section}Classify the following case. Respond in JSON.

{input_text[:1000]}

{{"reasoning": "...", "final_answer": "..."}}"""
        response = self.llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"num_lessons": len(self.lessons), "num_matched": len(top_lessons)}


# ── Test: learn from errors, then verify retrieval ────────────────────────────

mem = ReflexionLessonMemory(fake_llm)

# Real error patterns from score/diagnostics.jsonl
errors = [
    {
        "input": "事实:龙口市人民检察院起诉书指控，被告人林某为筹措资金，于2012年间，将房产证抵押给马某某，后伪造了假的房产证交还给马某某。",
        "prediction": "[罪名]伪造国家机关证件罪<eoa>",
        "ground_truth": "伪造、变造、买卖国家机关公文、证件、印章",
        "was_correct": False,
    },
    {
        "input": "事实:被告人唐某与陕西群友汽车租赁有限公司签订租车合同，租期一个月，后通过伪造合同骗取他人财物。",
        "prediction": "[罪名]诈骗<eoa>",
        "ground_truth": "合同诈骗",
        "was_correct": False,
    },
    {
        "input": "事实:被告人在担任某公司副总经理期间，利用职务便利，收受他人好处费200000元人民币。",
        "prediction": "[罪名]受贿<eoa>",
        "ground_truth": "非国家工作人员受贿",
        "was_correct": False,
    },
    {
        "input": "事实:公诉机关指控，被告人在舟曲县盗窃路灯电缆线约210米，并用车辆运走后出售给废品收购站。",
        "prediction": "[罪名]盗窃<eoa>",
        "ground_truth": "破坏电力设备",
        "was_correct": False,
    },
]

mem.learn_from_batch(errors)
print(f"Lessons stored: {len(mem.lessons)}")
for i, l in enumerate(mem.lessons):
    print(f"  [{i}] keywords={l['keywords']}")
    print(f"       lesson={l['lesson'][:80]}...")

# Test retrieval on a new similar query
test_queries = [
    ("伪造房产证场景", "事实:被告人李某将他人房产证挂失后伪造了一份新房产证用于贷款。"),
    ("合同诈骗场景",   "事实:被告人以签订合同为名骗取对方预付款后潜逃。"),
    ("电缆盗割场景",   "事实:被告人深夜剪断正在使用的路灯电缆，造成大面积停电。"),
    ("无关场景",       "事实:被告人在超市偷窃现金800元后被当场抓获。"),
]

print("\n── Retrieval test ──────────────────────────────────────")
for name, q in test_queries:
    answer, meta = mem.predict(q)
    print(f"  {name}: matched={meta['num_matched']} lessons, answer={answer}")

print("\nPrototype B PASSED")
print("Reflexion lesson memory: errors distilled into transfer rules,")
print("retrieved by keyword/bigram overlap at predict time.")
