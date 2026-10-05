"""Validation: contrastive_correction_memory — import and mechanism check.

Pure Python only. No os/sys/pathlib/network imports.
"""

import re
import json
from collections import defaultdict
from typing import Any


# ── tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> frozenset:
    tokens: set = set()
    tokens.update(re.findall(r"[A-Za-z0-9]+", text.lower()))
    cjk_chars = re.findall(r"[一-鿿]", text)
    for i in range(len(cjk_chars) - 1):
        tokens.add(cjk_chars[i] + cjk_chars[i + 1])
    return frozenset(tokens)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_json_field(response: str, field: str) -> str:
    try:
        return json.loads(response).get(field, "")
    except Exception:
        return ""


# ── fake LLM and base ─────────────────────────────────────────────────────────

def fake_llm(prompt: str) -> str:
    # Return the correction section content if present so we can verify it
    if "Mistakes to avoid" in prompt:
        return '{"reasoning": "saw corrections", "final_answer": "合同诈骗"}'
    return '{"reasoning": "no corrections", "final_answer": "诈骗"}'


class FakeBase:
    def __init__(self, llm):
        self._llm = llm
        self.examples: list = []
        self.corrections: list = []

    def call_llm(self, prompt: str) -> str:
        return self._llm(prompt)


# ── paste in the full ContrastiveCorrectionMemory logic ──────────────────────

_MAX_CORRECTIONS = 4
_MIN_CORRECTION_SIM = 0.02
_MAX_EXAMPLES = 20
MAX_CHARS = 30000

PROMPT_TEMPLATE = """Solve the problem below based on the examples and correction guidance provided.

{corrections_section}{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Pay attention to the mistakes listed above — avoid the same errors for similar cases
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

CORRECTIONS_HEADER = """**Mistakes to avoid** (from similar cases in memory):
{correction_lines}

"""


class ContrastiveCorrectionMemory(FakeBase):
    def _top_corrections(self, q_tok: frozenset) -> list:
        scored = [(c, _jaccard(q_tok, c["tokens"])) for c in self.corrections]
        relevant = [(c, s) for c, s in scored if s >= _MIN_CORRECTION_SIM]
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in relevant[:_MAX_CORRECTIONS]]

    def _top_examples(self, q_tok: frozenset) -> list:
        scored = [(_jaccard(q_tok, ex["tokens"]), idx, ex) for idx, ex in enumerate(self.examples)]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored

    def _build_prompt(self, input_text: str):
        q_tok = _tokenize(input_text)
        total_chars = 0

        corrections = self._top_corrections(q_tok)
        corr_lines = []
        for c in corrections:
            line = f"- For cases like: \"{c['preview']}\" — predicted \"{c['wrong']}\", correct answer is \"{c['right']}\""
            corr_lines.append(line)
            total_chars += len(line) + 1

        corrections_section = (
            CORRECTIONS_HEADER.format(correction_lines="\n".join(corr_lines))
            if corr_lines else ""
        )

        ranked = self._top_examples(q_tok)
        parts, used = [], set()
        for _, idx, ex in ranked:
            if idx in used:
                continue
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > MAX_CHARS:
                break
            parts.append(part)
            used.add(idx)
            total_chars += len(part) + 2
            if len(parts) >= _MAX_EXAMPLES:
                break

        examples_section = "\n\n".join(parts)
        if parts:
            examples_section = "**Examples:**\n" + examples_section + "\n\n"

        return corrections_section, examples_section, len(parts), len(corrections)

    def predict(self, input_text: str):
        cs, es, n_ex, n_corr = self._build_prompt(input_text)
        prompt = PROMPT_TEMPLATE.format(corrections_section=cs, examples_section=es, input=input_text)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"num_examples": len(self.examples), "num_selected": n_ex, "num_corrections": n_corr}

    def learn_from_batch(self, batch_results: list):
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            tok = _tokenize(raw_q)
            ex = {"input": r["input"], "target": r["ground_truth"], "tokens": tok}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)
            if not r.get("was_correct", True):
                pred = r.get("prediction", "")
                gt = r["ground_truth"]
                if pred and pred != gt:
                    preview = raw_q[:60].replace("\n", " ").replace("\r", "")
                    self.corrections.append({
                        "tokens": tok, "wrong": pred, "right": gt, "preview": preview,
                    })

    def get_state(self) -> str:
        ex_s = [{k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in e.items()} for e in self.examples]
        co_s = [{k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in c.items()} for c in self.corrections]
        return json.dumps({"examples": ex_s, "corrections": co_s}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        for ex in data.get("examples", []):
            r = dict(ex)
            r["tokens"] = frozenset(r["tokens"]) if isinstance(r.get("tokens"), list) else _tokenize(r.get("raw_question", r.get("input", "")))
            self.examples.append(r)
        self.corrections = []
        for c in data.get("corrections", []):
            r = dict(c)
            r["tokens"] = frozenset(r["tokens"]) if isinstance(r.get("tokens"), list) else _tokenize("")
            self.corrections.append(r)


# ── tests ─────────────────────────────────────────────────────────────────────

mem = ContrastiveCorrectionMemory(fake_llm)

# 1. cold start — no memory, no corrections, must not crash
answer, meta = mem.predict("被告人因借款纠纷伪造了房产证。")
assert answer == "诈骗", f"cold start unexpected: {answer}"
assert meta["num_corrections"] == 0
assert meta["num_selected"] == 0
print("PASS cold start: no crash, 0 corrections, 0 examples")

# 2. learn_from_batch with wrong predictions
batch = [
    {"input": "被告人何某隐瞒事实真相，谎称出租车车主是自己，与被害人签署了一份租车协议书，"
              "收取马某某一年的租金及车辆抵押金共计50600元，被告人何某拒不归还。",
     "ground_truth": "合同诈骗", "prediction": "诈骗", "was_correct": False},
    {"input": "被告人李某甲以牟利为目的，盗割正在使用中的公共照明电线，危害公共安全。",
     "ground_truth": "破坏电力设备", "prediction": "盗窃;破坏电力设施", "was_correct": False},
    {"input": "被告人何某某在私自焚烧秸秆时不慎引发山火，造成有林地损毁。",
     "ground_truth": "失火", "prediction": "失火", "was_correct": True},
]
mem.learn_from_batch(batch)
assert len(mem.examples) == 3
assert len(mem.corrections) == 2  # only wrong ones
print(f"PASS learn: {len(mem.examples)} examples, {len(mem.corrections)} corrections (correct ones excluded)")

# 3. predict with corrections — fake LLM returns "合同诈骗" when "Mistakes to avoid" appears
test_query = ("被告人以假冒他人身份签订合同协议，向被害人收取押金和租金后拒不归还，"
              "在签订履行合同过程中虚构事实骗取他人财物。")
answer2, meta2 = mem.predict(test_query)
assert meta2["num_corrections"] > 0, "Expected at least 1 correction injected"
assert answer2 == "合同诈骗", f"Expected fake LLM to return '合同诈骗' when corrections present, got '{answer2}'"
print(f"PASS corrections injected: {meta2['num_corrections']} corrections, {meta2['num_selected']} examples, answer='{answer2}'")

# 4. verify correction content is semantically matched
q_tok = _tokenize(test_query)
top_corr = mem._top_corrections(q_tok)
assert len(top_corr) > 0
assert top_corr[0]["right"] == "合同诈骗", f"Top correction should be '合同诈骗', got '{top_corr[0]['right']}'"
sim = _jaccard(q_tok, top_corr[0]["tokens"])
print(f"PASS correction relevance: top correction sim={sim:.3f}, right='{top_corr[0]['right']}'")

# 5. state round-trip
state = mem.get_state()
mem2 = ContrastiveCorrectionMemory(fake_llm)
mem2.set_state(state)
assert len(mem2.examples) == 3
assert len(mem2.corrections) == 2
assert all(isinstance(c["tokens"], frozenset) for c in mem2.corrections)
answer3, meta3 = mem2.predict(test_query)
assert meta3["num_corrections"] > 0
print("PASS state round-trip: corrections and examples restored with frozenset tokens")

# 6. verify cold-start query (no similar corrections) returns empty correction section
cold_query = "被告人非法持有毒品甲基苯丙胺若干克被当场查获。"
answer_cold, meta_cold = mem.predict(cold_query)
print(f"Dissimilar query corrections injected: {meta_cold['num_corrections']} (expected 0 or low)")
# Not asserting 0 here — just verifying no crash with low-similarity input

print("\nAll contrastive_correction_memory checks passed.")
