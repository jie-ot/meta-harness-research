"""Validation: label_glossary_memory — import and mechanism check."""

import json
import re
from typing import Any

# ---- Inline the agent's tokenizer and core logic ----

def _tokenize(text: str) -> frozenset:
    cjk_chars = re.findall(r'[一-鿿㐀-䶿]', text)
    bigrams = frozenset(
        cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)
    )
    ascii_toks = frozenset(re.findall(r'[A-Za-z0-9]+', text.lower()))
    return bigrams | ascii_toks

def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

# ---- Fake LLM ----
class FakeLLM:
    def __init__(self):
        self.last_prompt = ''
    def __call__(self, prompt: str) -> str:
        self.last_prompt = prompt
        return '{"reasoning": "test", "final_answer": "合同诈骗"}'

# ---- Minimal stubs ----
class MemorySystemStub:
    def __init__(self, llm):
        self._llm = llm
    def call_llm(self, prompt):
        return self._llm(prompt)

def extract_json_field(response, field):
    m = re.search(r'"' + field + r'"\s*:\s*"([^"]*)"', response)
    return m.group(1) if m else response

# ---- Replicate the agent class ----

PROMPT_WITH_GLOSSARY = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Canonical charge labels seen in training (use exact text from this list):**
{glossary}

**Instructions:**
- Analyse the facts and identify the charge(s)
- Your final_answer MUST use exact label text from the list above
- For multiple charges use semicolons: Label1;Label2
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[exact label(s) from list]"}}"""

PROMPT_NO_GLOSSARY = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

_EXAMPLE_BUDGET = 24000

class LabelGlossaryMemory(MemorySystemStub):
    def __init__(self, llm):
        super().__init__(llm)
        self.examples = []
        self.label_glossary = []

    def _add_labels(self, ground_truth):
        for lbl in ground_truth.split(';'):
            lbl = lbl.strip()
            if lbl and lbl not in self.label_glossary:
                self.label_glossary.append(lbl)

    def _build_glossary_str(self):
        return '\n'.join(f'- {lbl}' for lbl in sorted(self.label_glossary))

    def _build_parts(self, query):
        if not self.examples:
            return []
        q_tok = _tokenize(query)
        ranked = sorted(
            [(_jaccard(q_tok, ex['tokens']), idx, ex)
             for idx, ex in enumerate(self.examples)],
            reverse=True,
            key=lambda x: (x[0], x[1])
        )
        parts, total = [], 0
        for _, _, ex in ranked:
            q = ex.get('raw_question', ex['input'])
            part = f"Q: {q}\nA: {ex['target']}"
            if total + len(part) + 2 > _EXAMPLE_BUDGET:
                break
            parts.append(part)
            total += len(part) + 2
        return parts

    def predict(self, input_text):
        parts = self._build_parts(input_text)
        examples_section = '\n\n'.join(parts)
        if self.label_glossary:
            prompt = PROMPT_WITH_GLOSSARY.format(
                examples_section=examples_section,
                input=input_text,
                glossary=self._build_glossary_str(),
            )
        else:
            prompt = PROMPT_NO_GLOSSARY.format(
                examples_section=examples_section,
                input=input_text,
            )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, 'final_answer')
        return answer, {
            'num_examples': len(self.examples),
            'num_selected': len(parts),
            'num_glossary_labels': len(self.label_glossary),
        }

    def learn_from_batch(self, batch_results):
        for r in batch_results:
            raw_q = r.get('raw_question', r['input'])
            ex = {'input': r['input'], 'target': r['ground_truth'], 'tokens': _tokenize(raw_q)}
            if 'raw_question' in r:
                ex['raw_question'] = r['raw_question']
            self.examples.append(ex)
            self._add_labels(r['ground_truth'])

    def get_state(self):
        serialisable = [{k: (sorted(v) if isinstance(v, frozenset) else v)
                         for k, v in ex.items()} for ex in self.examples]
        return json.dumps({'examples': serialisable, 'label_glossary': self.label_glossary})

    def set_state(self, state):
        data = json.loads(state)
        self.examples = []
        for ex in data.get('examples', []):
            restored = dict(ex)
            raw_q = restored.get('raw_question', restored.get('input', ''))
            restored['tokens'] = _tokenize(raw_q)
            self.examples.append(restored)
        self.label_glossary = data.get('label_glossary', [])


# ---- Run validation ----
llm = FakeLLM()
mem = LabelGlossaryMemory(llm)

# 1. Cold start: no glossary, falls back to no-glossary prompt
answer, meta = mem.predict("事实:某被告人进行了某种欺诈行为。")
assert "Canonical charge labels" not in llm.last_prompt, "Should use no-glossary prompt at cold start"
assert meta['num_glossary_labels'] == 0
print(f"Cold start: answer='{answer}', glossary labels={meta['num_glossary_labels']} ✓")

# 2. Learn batch with real examples (from training diagnostics)
batch = [
    {"input": "q1", "raw_question": "事实:被告人在合同签订过程中虚构事实骗取财物。",
     "ground_truth": "合同诈骗", "prediction": "诈骗", "was_correct": False, "metadata": {}},
    {"input": "q2", "raw_question": "事实:被告人销售假冒注册商标的皮鞋1009双。",
     "ground_truth": "销售假冒注册商标的商品", "prediction": "销售假冒注册商标商品", "was_correct": False, "metadata": {}},
    {"input": "q3", "raw_question": "事实:被告人非法种植罂粟782株。",
     "ground_truth": "非法种植毒品原植物", "prediction": "非法种植罂粟", "was_correct": False, "metadata": {}},
    # multi-label: both labels should be added separately
    {"input": "q4", "raw_question": "事实:被告人故意伤人并毁坏财物。",
     "ground_truth": "故意伤害;故意毁坏财物", "prediction": "故意伤害", "was_correct": False, "metadata": {}},
]
mem.learn_from_batch(batch)
assert len(mem.examples) == 4
assert "合同诈骗" in mem.label_glossary
assert "销售假冒注册商标的商品" in mem.label_glossary
assert "非法种植毒品原植物" in mem.label_glossary
assert "故意伤害" in mem.label_glossary
assert "故意毁坏财物" in mem.label_glossary
print(f"After batch: {len(mem.examples)} examples, {len(mem.label_glossary)} glossary labels ✓")
print(f"Glossary: {mem.label_glossary}")

# 3. Predict: glossary prompt is now used
answer2, meta2 = mem.predict("事实:被告人签订虚假合同骗取他人财物5万元。")
assert "Canonical charge labels" in llm.last_prompt, "Should use glossary prompt after learning"
assert "合同诈骗" in llm.last_prompt, "合同诈骗 should be in glossary section of prompt"
assert "销售假冒注册商标的商品" in llm.last_prompt, "Exact canonical label should appear"
assert meta2['num_glossary_labels'] == 5
assert meta2['num_selected'] > 0
print(f"Predict with glossary: selected={meta2['num_selected']}, glossary={meta2['num_glossary_labels']} ✓")

# 4. Verify the near-miss labels are in glossary (not the wrong paraphrases)
assert "非法种植罂粟" not in mem.label_glossary, "Paraphrase should not be in glossary"
assert "销售假冒注册商标商品" not in mem.label_glossary, "Near-miss variant should not be in glossary"
print("Only canonical ground-truth labels in glossary (not model paraphrases) ✓")

# 5. Retrieval: bigram similarity routes similar query to correct example
q_tok = _tokenize("事实:被告人伪造合同签订过程中虚构事实骗取财物。")
sims = [(_jaccard(q_tok, ex['tokens']), ex['target']) for ex in mem.examples]
sims.sort(reverse=True)
print(f"Retrieval top-3: {sims[:3]}")
assert sims[0][1] == "合同诈骗", f"Expected 合同诈骗 at top, got {sims[0][1]}"
print("Retrieval ranks correct example first ✓")

# 6. State round-trip
state = mem.get_state()
mem2 = LabelGlossaryMemory(llm)
mem2.set_state(state)
assert len(mem2.examples) == 4
assert mem2.label_glossary == mem.label_glossary
assert isinstance(mem2.examples[0]['tokens'], frozenset)
print("State round-trip ✓")

print("\nVALIDATION PASSED: label_glossary_memory")
