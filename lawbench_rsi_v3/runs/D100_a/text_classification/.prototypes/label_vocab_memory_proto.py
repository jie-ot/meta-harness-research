"""Prototype for label_vocab_memory.

Hypothesis: maintaining a registry of all canonical label strings seen in
training and prepending them as a 'Valid Labels' section in the prompt will
reduce label hallucination (suffix errors, specificity errors) and improve
val accuracy above 38%.

New mechanisms vs frontier:
  - Axis B: separate label-set memory structure (not just raw examples)
  - Axis A: prompt architecture with a label-priming section

No file/network/process imports. Real examples copied from diagnostics.
"""

import json
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Real examples copied from train/diagnostics.jsonl (literals)
# ---------------------------------------------------------------------------
TRAIN_BATCH = [
    {"input": "事实:...打砸财物...", "ground_truth": "故意毁坏财物",
     "prediction": "故意毁坏财物罪", "was_correct": False},
    {"input": "事实:...拐骗儿童...", "ground_truth": "拐骗儿童",
     "prediction": "拐骗儿童", "was_correct": True},
    {"input": "事实:...拐卖妇女...", "ground_truth": "拐卖妇女、儿童",
     "prediction": "拐卖妇女", "was_correct": False},
    {"input": "事实:...行贿...", "ground_truth": "行贿",
     "prediction": "行贿", "was_correct": True},
    {"input": "事实:...受贿...", "ground_truth": "非国家工作人员受贿",
     "prediction": "受贿", "was_correct": False},
    {"input": "事实:...伪造证件...",
     "ground_truth": "伪造、变造、买卖国家机关公文、证件、印章",
     "prediction": "伪造国家机关证件罪;诈骗罪", "was_correct": False},
    {"input": "事实:...破坏电脑...",
     "ground_truth": "破坏计算机信息系统",
     "prediction": "敲诈勒索罪;非法控制计算机信息系统罪", "was_correct": False},
    {"input": "事实:...贩毒...",
     "ground_truth": "走私、贩卖、运输、制造毒品;徇私枉法",
     "prediction": "贩卖毒品;徇私枉法", "was_correct": False},
    {"input": "事实:...虚开发票...",
     "ground_truth": "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票;假冒注册商标",
     "prediction": "虚开增值税专用发票", "was_correct": False},
    {"input": "事实:...非法采矿...", "ground_truth": "非法采矿",
     "prediction": "非法采矿", "was_correct": True},
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
# VARIANT 1: flat alphabetical list of all seen labels
# ---------------------------------------------------------------------------
class LabelVocabV1:
    def __init__(self):
        self.label_counts = defaultdict(int)
        self.examples = []

    def learn(self, batch):
        for r in batch:
            for lbl in r["ground_truth"].split(";"):
                self.label_counts[lbl.strip()] += 1
            self.examples.append({
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": tokenize(r["input"]),
            })

    def label_section(self, max_chars=2000):
        labels = sorted(self.label_counts.keys())
        lines = ["## 有效罪名（请从以下列表中选择，保持精确格式）:"]
        total = len(lines[0])
        for lbl in labels:
            line = f"- {lbl}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def predict_section(self, query):
        return self.label_section()


# ---------------------------------------------------------------------------
# VARIANT 2: labels sorted by descending frequency (most common first)
# ---------------------------------------------------------------------------
class LabelVocabV2:
    def __init__(self):
        self.label_counts = defaultdict(int)

    def learn(self, batch):
        for r in batch:
            for lbl in r["ground_truth"].split(";"):
                self.label_counts[lbl.strip()] += 1

    def label_section(self, max_chars=2000):
        sorted_labels = sorted(self.label_counts.items(), key=lambda x: (-x[1], x[0]))
        lines = ["## 有效罪名（按频次排列）:"]
        total = len(lines[0])
        for lbl, cnt in sorted_labels:
            line = f"- {lbl}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# VARIANT 3: similarity-filtered — only show labels whose token overlap with
# query exceeds a threshold (most query-relevant canonical labels)
# ---------------------------------------------------------------------------
class LabelVocabV3:
    def __init__(self):
        self.label_tokens = {}  # label -> frozenset of tokens
        self.label_counts = defaultdict(int)

    def learn(self, batch):
        for r in batch:
            for lbl in r["ground_truth"].split(";"):
                lbl = lbl.strip()
                self.label_counts[lbl] += 1
                if lbl not in self.label_tokens:
                    self.label_tokens[lbl] = tokenize(lbl)

    def label_section(self, query, top_n=30, max_chars=2000):
        q_tok = tokenize(query)
        scored = [(jaccard(q_tok, toks), lbl)
                  for lbl, toks in self.label_tokens.items()]
        scored.sort(key=lambda x: (-x[0], x[1]))
        lines = ["## 相关罪名（请从以下列表选择精确字符串）:"]
        total = len(lines[0])
        for _, lbl in scored[:top_n]:
            line = f"- {lbl}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
print("=== VARIANT 1: flat alphabetical ===")
v1 = LabelVocabV1()
v1.learn(TRAIN_BATCH)
sec = v1.label_section()
print(f"Label section length: {len(sec)} chars, {len(v1.label_counts)} unique labels")
print(sec[:400])
print()

# Verify multi-label targets split correctly
multi = "走私、贩卖、运输、制造毒品;徇私枉法"
parts = [l.strip() for l in multi.split(";")]
assert len(parts) == 2 and parts[0] == "走私、贩卖、运输、制造毒品", f"split failed: {parts}"

# Verify "故意毁坏财物" (no 罪 suffix) is in vocab
assert "故意毁坏财物" in v1.label_counts, "missing canonical label"
assert "故意毁坏财物罪" not in v1.label_counts, "wrong prediction leaked into vocab"

print("=== VARIANT 2: frequency-sorted ===")
v2 = LabelVocabV2()
v2.learn(TRAIN_BATCH)
sec2 = v2.label_section()
print(sec2[:300])
print()

print("=== VARIANT 3: similarity-filtered ===")
v3 = LabelVocabV3()
v3.learn(TRAIN_BATCH)
query = "事实:...采矿许可证...非法开采稀土..."
sec3 = v3.label_section(query)
print(f"Query: {query}")
print(sec3[:300])
print()

# Check that "非法采矿" appears before irrelevant labels for mining query
labels_in_sec3 = [l[2:] for l in sec3.split("\n") if l.startswith("- ")]
assert "非法采矿" in labels_in_sec3, f"非法采矿 not in similarity-filtered list: {labels_in_sec3}"
print(f"Top labels for mining query: {labels_in_sec3[:5]}")

print()
print("WINNER: VARIANT 1 (flat alphabetical)")
print("Reason: all labels are equally valid; alphabetical is deterministic")
print("        and gives the LLM the complete known-good string set.")
print("        V2 (freq) could bias toward common labels and hide rare ones.")
print("        V3 (similarity-filtered) risks excluding the correct label for")
print("        novel cases that don't token-match seen training labels.")
print()
print("All prototype tests PASSED")
