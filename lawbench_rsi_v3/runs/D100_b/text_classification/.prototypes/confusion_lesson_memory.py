"""Prototype: Confusion-Lesson Memory (Candidate A, axis F).

Mechanism: after a wrong prediction the system calls the LLM once to distill a
short corrective lesson ("when you see pattern X, prefer Y over Z because …").
Lessons are stored alongside raw examples.  At predict time the most relevant
lessons are injected into a dedicated "Common mistakes" section before the
question, giving the LLM an explicit advisory that raw examples cannot.

This prototype uses a fake LLM so no real API calls are made.
"""

import re
import json

# ---------------------------------------------------------------------------
# Minimal stubs (replicate what memory_system.py provides at runtime)
# ---------------------------------------------------------------------------

def _tokenize(text: str):
    cjk_chars = re.findall(r'[一-鿿㐀-䶿]', text)
    bigrams = frozenset(
        cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)
    )
    ascii_toks = frozenset(re.findall(r'[A-Za-z0-9]+', text.lower()))
    return bigrams | ascii_toks

def _jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

# ---------------------------------------------------------------------------
# Fake LLM — returns canned lesson or answer strings
# ---------------------------------------------------------------------------

LESSON_RESPONSES = {
    "行贿": (
        '{"lesson": "When the defendant gives money to a state official to obtain '
        'benefits, prefer 行贿 (bribery by the giver). '
        '非国家工作人员受贿 applies when the recipient is NOT a state employee."}'
    ),
    "故意伤害": (
        '{"lesson": "When facts describe both physical assault causing injury AND '
        'property damage or mob conduct, emit both charges separated by semicolon. '
        'Do not drop the secondary charge."}'
    ),
}

def fake_llm(prompt: str) -> str:
    for key, resp in LESSON_RESPONSES.items():
        if key in prompt:
            return resp
    return '{"lesson": "Check facts carefully before choosing a single charge."}'

# ---------------------------------------------------------------------------
# Simplified ConfusionLessonMemory logic
# ---------------------------------------------------------------------------

class ConfusionLessonMemoryProto:
    def __init__(self):
        self.examples = []       # {input, target, tokens}
        self.lessons = []        # {tokens, lesson_text}
        self.label_glossary = []

    def _add_labels(self, gt):
        for lbl in gt.split(';'):
            lbl = lbl.strip()
            if lbl and lbl not in self.label_glossary:
                self.label_glossary.append(lbl)

    # --- lesson retrieval ---
    def _top_lessons(self, query_toks, k=3):
        ranked = sorted(
            self.lessons,
            key=lambda l: _jaccard(query_toks, l['tokens']),
            reverse=True,
        )
        return ranked[:k]

    # --- example retrieval (plain Jaccard, budget-capped) ---
    def _top_examples(self, query_toks, budget=18000):
        ranked = sorted(
            self.examples,
            key=lambda e: _jaccard(query_toks, e['tokens']),
            reverse=True,
        )
        parts, total = [], 0
        for ex in ranked:
            part = f"Q: {ex['input'][:200]}\nA: {ex['target']}"
            if total + len(part) + 2 > budget:
                break
            parts.append(part)
            total += len(part) + 2
        return parts

    def predict(self, inp: str) -> str:
        q_tok = _tokenize(inp)
        lessons = self._top_lessons(q_tok)
        examples = self._top_examples(q_tok)

        lesson_section = ""
        if lessons:
            lines = "\n".join(f"- {l['lesson_text']}" for l in lessons)
            lesson_section = f"\n\n**Common mistakes to avoid:**\n{lines}"

        glossary = '\n'.join(f'- {lbl}' for lbl in sorted(self.label_glossary))
        prompt = (
            '\n\n'.join(examples)
            + lesson_section
            + f"\n\n**Problem:**\n{inp}"
            + (f"\n\n**Canonical labels:**\n{glossary}" if glossary else "")
            + '\n\n{"reasoning": "...", "final_answer": "..."}'
        )
        # return full prompt for testing (production returns LLM answer)
        return prompt

    def learn_from_batch(self, batch):
        for r in batch:
            raw_q = r.get('raw_question', r['input'])
            toks = _tokenize(raw_q)
            self.examples.append({'input': raw_q, 'target': r['ground_truth'], 'tokens': toks})
            self._add_labels(r['ground_truth'])

            if not r['was_correct']:
                # Ask LLM for a corrective lesson
                lesson_prompt = (
                    f"The model predicted «{r['prediction']}» but the correct answer is «{r['ground_truth']}».\n"
                    f"Write a one-sentence lesson for future predictions. "
                    f"Respond: {{\"lesson\": \"<lesson>\"}}"
                )
                resp = fake_llm(lesson_prompt)
                lesson_text = ""
                try:
                    lesson_text = json.loads(resp).get('lesson', '')
                except Exception:
                    pass
                if lesson_text:
                    self.lessons.append({'tokens': toks, 'lesson_text': lesson_text})

# ---------------------------------------------------------------------------
# Test with real examples from training diagnostics
# ---------------------------------------------------------------------------

TRAIN_SAMPLES = [
    # item lawbench_3-3_0215: predicted 非国家工作人员受贿, target 行贿
    {
        'input': '被告人黄某某…找到于某甲帮忙提高拆迁补偿款…将3万元送给于某甲。',
        'raw_question': '被告人黄某某…找到于某甲帮忙提高拆迁补偿款…将3万元送给于某甲。',
        'prediction': '非国家工作人员受贿',
        'ground_truth': '行贿',
        'was_correct': False,
    },
    # item lawbench_3-3_0004: predicted 故意毁坏财物, target 故意伤害;故意毁坏财物
    {
        'input': '宋某持钢管殴打陈某甲，后打砸店铺内冰箱电脑等财物价值2609元。经鉴定陈某甲重伤六级。',
        'raw_question': '宋某持钢管殴打陈某甲，后打砸店铺内冰箱电脑等财物价值2609元。经鉴定陈某甲重伤六级。',
        'prediction': '故意毁坏财物',
        'ground_truth': '故意伤害;故意毁坏财物',
        'was_correct': False,
    },
    # correct case — should not generate a lesson
    {
        'input': '被告人朱某某通过虚假充值，致被害单位损失人民币1316910元。',
        'raw_question': '被告人朱某某通过虚假充值，致被害单位损失人民币1316910元。',
        'prediction': '破坏计算机信息系统',
        'ground_truth': '破坏计算机信息系统',
        'was_correct': True,
    },
]

def test_lesson_memory():
    print("=== ConfusionLessonMemory Prototype ===\n")
    mem = ConfusionLessonMemoryProto()

    # cold start
    out = mem.predict("被告人李某某带领十余人阻止施工，持顶托将刘B打伤，刘A刘B均轻伤。")
    print(f"[cold start] prompt prefix: {out}")
    assert "Problem" in out, "prompt missing Problem section"

    # learn
    mem.learn_from_batch(TRAIN_SAMPLES)
    print(f"\nAfter learning:")
    print(f"  examples stored: {len(mem.examples)}")
    print(f"  lessons stored:  {len(mem.lessons)}")
    print(f"  glossary size:   {len(mem.label_glossary)}")
    assert len(mem.lessons) == 2, f"Expected 2 lessons for 2 wrong items, got {len(mem.lessons)}"

    # predict with lessons available
    query = "被告人宋某持钢管殴打被害人并打砸店铺财物。"
    q_tok = _tokenize(query)
    top = mem._top_lessons(q_tok, k=2)
    print(f"\n  top lessons for assault+property query:")
    for l in top:
        print(f"    • {l['lesson_text'][:80]}")
    assert len(top) > 0

    # variant: lesson injection raises Jaccard for relevant query
    lesson_toks = [_jaccard(q_tok, l['tokens']) for l in mem.lessons]
    print(f"  lesson Jaccard scores: {[round(s,3) for s in lesson_toks]}")

    # test that correct items don't add lessons
    all_lesson_texts = [l['lesson_text'] for l in mem.lessons]
    assert not any('破坏计算机' in t for t in all_lesson_texts), \
        "Correct item should not produce a lesson"

    out2 = mem.predict(query)
    print(f"\n[warm] prompt prefix: {out2}")
    assert "Common mistakes" in out2, "Lesson section missing from warm prompt"

    print("\n✓ All assertions passed\n")

    # variant comparison: with vs without lesson section
    print("--- Variant comparison ---")
    print("Without lesson section: model sees only examples → may repeat prior confusion")
    print("With lesson section:    model sees explicit 'prefer 行贿 when giver gives money'")
    print("→ lesson injection is the novel mechanism; raw examples cannot encode this rule")

if __name__ == '__main__':
    test_lesson_memory()
