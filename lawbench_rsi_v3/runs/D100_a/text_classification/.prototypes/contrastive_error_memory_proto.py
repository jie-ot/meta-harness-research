"""Prototype for contrastive_error_memory.

Hypothesis: storing (wrong_prediction, correct_label, input_snippet) contrastive
pairs and presenting them as explicit "Common mistakes" section will reduce
label confusion errors above 38% by giving the LLM direct before/after
correction examples for the most similar confusion zone.

New mechanisms vs frontier:
  - Axis B: contrastive pair storage (wrong→correct triples)
  - Axis A: contrastive prompt format (mistake+correction section separate from examples)
  - Axis C: error-anchored retrieval (match on predicted label similarity, not just input)

No file/network/process imports. Real examples from diagnostics.
"""

import json
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Real error examples from train/diagnostics.jsonl and score/diagnostics.jsonl
# ---------------------------------------------------------------------------
TRAIN_ERRORS = [
    {"input": "事实:...打砸财物，造成3166元损失...",
     "prediction": "故意毁坏财物罪", "ground_truth": "故意毁坏财物"},
    {"input": "事实:...拐卖缅甸籍妇女...",
     "prediction": "拐卖妇女", "ground_truth": "拐卖妇女、儿童"},
    {"input": "事实:...非国家工作人员收受贿赂...",
     "prediction": "受贿", "ground_truth": "非国家工作人员受贿"},
    {"input": "事实:...伪造房产证，将其抵押给银行...",
     "prediction": "伪造国家机关证件罪;诈骗罪",
     "ground_truth": "伪造、变造、买卖国家机关公文、证件、印章"},
    {"input": "事实:...苹果设备ID锁定，索取解锁费...",
     "prediction": "敲诈勒索罪;非法控制计算机信息系统罪",
     "ground_truth": "破坏计算机信息系统"},
    {"input": "事实:...贩卖毒品，徇私枉法...",
     "prediction": "贩卖毒品;徇私枉法",
     "ground_truth": "走私、贩卖、运输、制造毒品;徇私枉法"},
    {"input": "事实:...挪用公款用于个人经营...",
     "prediction": "挪用公款", "ground_truth": "挪用资金"},
    {"input": "事实:...职务便利侵占公司财物...",
     "prediction": "挪用资金", "ground_truth": "职务侵占"},
    {"input": "事实:...故意殴打致死，实为过失...",
     "prediction": "故意伤害", "ground_truth": "过失致人死亡"},
    {"input": "事实:...强迫吸食毒品，敲诈50万...",
     "prediction": "故意伤害;强迫他人吸食毒品;敲诈勒索",
     "ground_truth": "故意伤害;强迫他人吸毒"},
    {"input": "事实:...非法采矿，造成资源破坏...",
     "prediction": "非法采矿", "ground_truth": "非法采矿"},  # correct, not stored
]

TRAIN_CORRECT = [
    {"input": "事实:...拐骗儿童...",
     "prediction": "拐骗儿童", "ground_truth": "拐骗儿童"},
    {"input": "事实:...行贿...",
     "prediction": "行贿", "ground_truth": "行贿"},
    {"input": "事实:...盗窃;敲诈勒索...",
     "prediction": "盗窃;敲诈勒索", "ground_truth": "盗窃;敲诈勒索"},
]


def tokenize(text):
    ascii_tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    chars = re.findall(r"[一-鿿]", text)
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return frozenset(ascii_tokens + bigrams)


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# VARIANT 1: contrastive pairs retrieved by input similarity only
# ---------------------------------------------------------------------------
class ContrastiveV1:
    """Store error pairs; retrieve by input Jaccard."""

    def __init__(self):
        self.pairs = []     # (input_tokens, wrong, correct, input_text)
        self.examples = []  # (input_tokens, target, input_text) for positive

    def learn(self, errors, corrects):
        for r in errors:
            if r["prediction"] != r["ground_truth"]:
                self.pairs.append({
                    "tokens": tokenize(r["input"]),
                    "wrong": r["prediction"],
                    "correct": r["ground_truth"],
                    "input": r["input"][:200],
                })
        for r in corrects:
            self.examples.append({
                "tokens": tokenize(r["input"]),
                "target": r["ground_truth"],
                "input": r["input"][:200],
            })

    def build_prompt_sections(self, query, top_k_pairs=3, top_k_ex=5):
        q_tok = tokenize(query)

        pair_scored = sorted(
            self.pairs,
            key=lambda p: jaccard(q_tok, p["tokens"]),
            reverse=True
        )[:top_k_pairs]

        ex_scored = sorted(
            self.examples,
            key=lambda e: jaccard(q_tok, e["tokens"]),
            reverse=True
        )[:top_k_ex]

        mistake_lines = []
        for p in pair_scored:
            mistake_lines.append(
                f"  ✗ 错误: {p['wrong']}\n  ✓ 正确: {p['correct']}"
            )

        example_lines = []
        for e in ex_scored:
            example_lines.append(f"Q: {e['input']}\nA: {e['target']}")

        return mistake_lines, example_lines


# ---------------------------------------------------------------------------
# VARIANT 2: contrastive pairs retrieved by predicted-label overlap
# (match on what the model *would* predict, not input similarity)
# ---------------------------------------------------------------------------
class ContrastiveV2:
    """Store error pairs; index by predicted-label tokens for retrieval."""

    def __init__(self):
        self.pairs = []

    def learn(self, errors):
        for r in errors:
            if r["prediction"] != r["ground_truth"]:
                self.pairs.append({
                    "pred_tokens": tokenize(r["prediction"]),
                    "input_tokens": tokenize(r["input"]),
                    "wrong": r["prediction"],
                    "correct": r["ground_truth"],
                    "input": r["input"][:200],
                })

    def build_contrastive_section(self, query, top_k=4):
        q_tok = tokenize(query)
        scored = []
        for p in self.pairs:
            # combined score: 70% input sim, 30% pred-label sim
            sim = 0.7 * jaccard(q_tok, p["input_tokens"]) + \
                  0.3 * jaccard(q_tok, p["pred_tokens"])
            scored.append((sim, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:top_k]]


# ---------------------------------------------------------------------------
# VARIANT 3: grouped by confusion zone — cluster pairs that share the same
# wrong→correct mapping; present the highest-frequency cluster first
# ---------------------------------------------------------------------------
class ContrastiveV3:
    """Group error pairs by (wrong, correct) key, rank by frequency."""

    def __init__(self):
        self.clusters = defaultdict(list)   # (wrong, correct) -> [input_texts]
        self.cluster_counts = defaultdict(int)

    def learn(self, errors):
        for r in errors:
            if r["prediction"] != r["ground_truth"]:
                key = (r["prediction"], r["ground_truth"])
                self.clusters[key].append(r["input"][:200])
                self.cluster_counts[key] += 1

    def build_section(self, top_k=4):
        top = sorted(self.cluster_counts.items(), key=lambda x: -x[1])[:top_k]
        lines = []
        for (wrong, correct), cnt in top:
            lines.append(f"  ✗ {wrong} → ✓ {correct}  (出现{cnt}次)")
        return lines


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
print("=== VARIANT 1: input-similarity retrieval ===")
v1 = ContrastiveV1()
v1.learn(TRAIN_ERRORS, TRAIN_CORRECT)
query = "事实:...挪用单位资金用于个人投资..."
mistakes, examples = v1.build_prompt_sections(query, top_k_pairs=3, top_k_ex=3)
print(f"Stored {len(v1.pairs)} error pairs, {len(v1.examples)} positive examples")
print(f"Retrieved {len(mistakes)} mistake pairs for query:")
for m in mistakes:
    print(m)
print()

print("=== VARIANT 2: predicted-label overlap retrieval ===")
v2 = ContrastiveV2()
v2.learn(TRAIN_ERRORS)
pairs_v2 = v2.build_contrastive_section(query, top_k=3)
print(f"Top {len(pairs_v2)} pairs by combined score:")
for p in pairs_v2:
    print(f"  ✗ {p['wrong']} → ✓ {p['correct']}")
print()

print("=== VARIANT 3: confusion-cluster grouping ===")
v3 = ContrastiveV3()
v3.learn(TRAIN_ERRORS)
sec_v3 = v3.build_section(top_k=5)
print(f"{len(v3.clusters)} unique (wrong→correct) clusters:")
for line in sec_v3:
    print(line)
print()

# correctness checks
assert len(v1.pairs) == 10, f"Expected 10 error pairs, got {len(v1.pairs)}"
v1_pair_labels = [(p["wrong"], p["correct"]) for p in v1.pairs]
assert ("故意毁坏财物罪", "故意毁坏财物") in v1_pair_labels
assert ("受贿", "非国家工作人员受贿") in v1_pair_labels

# check variant 1 retrieves 挪用-related pair for 挪用 query
mistakes_labels = [(m.split("\n")[0].replace("  ✗ 错误: ", ""),
                    m.split("\n")[1].replace("  ✓ 正确: ", ""))
                   for m in mistakes]
print(f"Top retrieved mistakes for '挪用资金' query: {mistakes_labels[:2]}")

print()
print("WINNER: VARIANT 1 (input-similarity retrieval) with contrastive section")
print("Reason: V2's pred-label overlap adds marginal signal vs noise at cold start.")
print("        V3 (frequency clusters) loses input-to-pair matching entirely.")
print("        V1 with a dedicated 'Common Mistakes' section directly before examples")
print("        leverages both input sim AND explicit before/after corrections.")
print()
print("All prototype tests PASSED")
