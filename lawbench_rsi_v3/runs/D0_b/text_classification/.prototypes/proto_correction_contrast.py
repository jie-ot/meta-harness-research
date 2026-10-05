"""Prototype: Contrastive correction memory.

Mechanism: store explicit (wrong_prediction → correct_label, query) triples
as "correction records". At predict time, find corrections whose query matches
the current input and inject a "Mistakes to avoid" section BEFORE the examples.

The model sees what predictions are wrong for similar cases, steering it away
from the most common error modes. This changes both prompt architecture (A)
and memory content (B).

No file/network I/O. Literals from score/diagnostics.jsonl.
"""

import re
from collections import defaultdict
from typing import Any


# ── tokenizer (bigram, reused from Candidate A) ─────────────────────────────

def _tokenize(text: str) -> frozenset:
    tokens = set()
    tokens.update(re.findall(r"[A-Za-z0-9]+", text.lower()))
    cjk = re.findall(r"[一-鿿]", text)
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i] + cjk[i + 1])
    return frozenset(tokens)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── real examples from score/diagnostics.jsonl ──────────────────────────────

EXAMPLES = [
    # (input_text, ground_truth, prediction, was_correct)
    (
        "被告单位通辽市某商业广场有限公司及其法定代表人沙某某，利用另案犯罪嫌疑人张某某的职务便利，"
        "以通辽市政府办公厅的名义出具了推荐函，为感谢张某某而向其支付了不正当报酬20万元。",
        "单位行贿", "行贿;非法采矿", False
    ),
    (
        "被告人何某隐瞒事实真相，谎称出租车车主是自己，与被害人马某某签署了一份租车协议书，"
        "收取马某某一年的租金及车辆抵押金共计50600元，被告人何某拒不归还。",
        "合同诈骗", "诈骗", False
    ),
    (
        "被告人李某甲以牟利为目的，盗割正在使用中的公共照明电线，危害公共安全。",
        "破坏电力设备", "盗窃;破坏电力设施", False
    ),
    (
        "被告人叶某甲从青田县季宅乡黄放口村摘取几个罂粟果，将罂粟果种植在自家农田中，"
        "经现场清点被铲除的已结果的罂粟植株共计782株。",
        "非法种植毒品原植物", "非法种植罂粟", False
    ),
    (
        "被告人何某某在三台县建设镇干坝王村私自焚烧秸秆不慎引发山火，"
        "造成的过火有林地面积为2.0071公顷。",
        "失火", "失火", True
    ),
    (
        "被告人郭某受雇于同案人，在广州富力城内销售假冒GIVENCHY等注册商标的皮鞋1009双，"
        "价值人民币6525350元。",
        "销售假冒注册商标的商品", "侵犯注册商标专用权", False
    ),
    (
        "被告人受贿，在收费员协助下，将药品销售给医院医生，支付不正当利益。",
        "对非国家工作人员行贿;对单位行贿", "行贿", False
    ),
]


# ── memory structures ─────────────────────────────────────────────────────────

class CorrectionMemory:
    def __init__(self):
        self.examples: list[dict] = []          # (input, target, tokens)
        self.corrections: list[dict] = []       # (input, wrong_pred, right_label, tokens)

    def learn(self, batch):
        for r in batch:
            tok = _tokenize(r["input"])
            self.examples.append({"input": r["input"], "target": r["ground_truth"], "tokens": tok})
            if not r["was_correct"]:
                self.corrections.append({
                    "input": r["input"],
                    "wrong": r["prediction"],
                    "right": r["ground_truth"],
                    "tokens": tok,
                })

    def _top_corrections(self, query: str, k: int = 3) -> list[dict]:
        q = _tokenize(query)
        scored = sorted(self.corrections, key=lambda c: _jaccard(q, c["tokens"]), reverse=True)
        return scored[:k]

    def _top_examples(self, query: str, k: int = 5) -> list[dict]:
        q = _tokenize(query)
        scored = sorted(self.examples, key=lambda e: _jaccard(q, e["tokens"]), reverse=True)
        return scored[:k]

    def build_prompt_sections(self, query: str) -> tuple[str, str]:
        corrections = self._top_corrections(query, k=3)
        examples = self._top_examples(query, k=5)

        corr_lines = []
        for c in corrections:
            if _jaccard(_tokenize(query), c["tokens"]) > 0.01:
                corr_lines.append(f"  Input like: {c['input'][:60]}...")
                corr_lines.append(f"  Wrong: {c['wrong']}  →  Correct: {c['right']}")
        correction_section = "\n".join(corr_lines) if corr_lines else "(none yet)"

        ex_lines = [f"Q: {e['input'][:80]}...\nA: {e['target']}" for e in examples]
        example_section = "\n\n".join(ex_lines)

        return correction_section, example_section


# ── test the mechanism ────────────────────────────────────────────────────────

mem = CorrectionMemory()

# Simulate training on first 5 examples, then predict on something similar
train_batch = [
    {"input": ex[0], "ground_truth": ex[1], "prediction": ex[2], "was_correct": ex[3]}
    for ex in EXAMPLES[:5]
]
mem.learn(train_batch)

# Test query: similar to the "合同诈骗" case
test_query = (
    "被告人以假冒他人身份签订协议，向被害人收取押金和租金，后拒不归还，"
    "在签订履行合同过程中采取欺诈手段骗取他人财物。"
)

corr_section, ex_section = mem.build_prompt_sections(test_query)

print("=== CORRECTION SECTION (injected before examples) ===")
print(corr_section)
print()
print("=== TOP EXAMPLES ===")
print(ex_section[:500] + "...")
print()

# Verify corrections are retrieved and match the right semantic clusters
corrections = mem._top_corrections(test_query, k=3)
print("Top corrections retrieved:")
for c in corrections:
    sim = _jaccard(_tokenize(test_query), c["tokens"])
    print(f"  sim={sim:.3f}  wrong={c['wrong'][:20]:22s}  right={c['right'][:20]}")

print()
print("=== VARIANT: what if corrections section is empty (cold start)? ===")
empty_mem = CorrectionMemory()
cs, es = empty_mem.build_prompt_sections(test_query)
print(f"Correction section: '{cs}'  (graceful empty)")
print(f"Example section: '{es}'  (empty pool, graceful)")

print()
print("=== VARIANT: correction ordering — by recency vs similarity ===")
# Check whether recency would be better
def top_corrections_recent(corrections_pool, query, k=3):
    """Most recent k corrections."""
    return corrections_pool[-k:]

sim_corrections = mem._top_corrections(test_query, k=3)
rec_corrections = top_corrections_recent(mem.corrections, test_query, k=3)

print("Similarity-ranked corrections:")
for c in sim_corrections:
    sim = _jaccard(_tokenize(test_query), c["tokens"])
    print(f"  sim={sim:.3f}  right={c['right']}")
print("Recency-ranked corrections:")
for c in rec_corrections:
    sim = _jaccard(_tokenize(test_query), c["tokens"])
    print(f"  sim={sim:.3f}  right={c['right']}")

print()
print("Conclusion: similarity-ranked corrections are more relevant. Proceeding with that.")
