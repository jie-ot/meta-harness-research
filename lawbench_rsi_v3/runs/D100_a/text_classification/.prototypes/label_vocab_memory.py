"""Validation prototype for label_vocab_memory.py — tests imports and mechanism."""

import re
import json
from collections import defaultdict

# ---------------------------------------------------------------------------
# Inline stubs for ..memory_system and ..llm (no file I/O)
# ---------------------------------------------------------------------------
from typing import Any
from abc import ABC, abstractmethod

LLMCallable = Any

def extract_json_field(response: str, field: str) -> str:
    m = re.search(r'"' + field + r'"\s*:\s*"([^"]*)"', response)
    return m.group(1) if m else response.strip()

class MemorySystem(ABC):
    def __init__(self, llm):
        self._llm = llm
    def call_llm(self, prompt: str) -> str:
        return self._llm(prompt)
    @abstractmethod
    def predict(self, input: str): ...
    @abstractmethod
    def learn_from_batch(self, batch_results): ...
    @abstractmethod
    def get_state(self) -> str: ...
    @abstractmethod
    def set_state(self, state: str) -> None: ...

# ---------------------------------------------------------------------------
# Copy the implementation inline (same logic as agents/label_vocab_memory.py)
# ---------------------------------------------------------------------------
_LABEL_VOCAB_MAX_CHARS = 4000
_EXAMPLES_MAX_CHARS = 26000
_TOP_CANDIDATE_LABELS = 3
_DISAMBIG_PER_CONFUSION = 2

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{label_section}

{examples_section}

**Problem:**
{input}

**Instructions:**
- You MUST choose label strings exactly from the Valid Labels list above
- Multiple labels are separated by semicolons
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""


def _tokenize(text):
    ascii_tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    chars = re.findall(r"[一-鿿]", text)
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return frozenset(ascii_tokens + bigrams)


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class LabelVocabMemory(MemorySystem):
    def __init__(self, llm):
        super().__init__(llm)
        self.examples = []
        self.confusion = defaultdict(lambda: defaultdict(int))
        self.label_vocab = defaultdict(int)

    def _label_section(self):
        if not self.label_vocab:
            return ""
        labels = sorted(self.label_vocab.keys())
        lines = ["## 有效罪名（请从以下列表中选择，保持精确格式）:"]
        total = len(lines[0])
        for lbl in labels:
            line = f"- {lbl}"
            if total + len(line) + 1 > _LABEL_VOCAB_MAX_CHARS:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def _scored(self, query):
        q_tok = _tokenize(query)
        result = [(_jaccard(q_tok, ex["tokens"]), idx, ex) for idx, ex in enumerate(self.examples)]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _confusion_targets_for(self, labels):
        label_set = set(labels)
        counts = defaultdict(int)
        for lbl in labels:
            row = self.confusion.get(lbl, {})
            for actual, cnt in row.items():
                if actual not in label_set:
                    counts[actual] += cnt
        return [lbl for lbl, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    def _build_parts(self, query):
        if not self.examples:
            return []
        ranked = self._scored(query)
        top_labels = [ex["target"] for _, _, ex in ranked[:_TOP_CANDIDATE_LABELS]]
        confused_with = self._confusion_targets_for(top_labels)
        parts = []
        used_indices = set()
        total_chars = 0

        def try_add(idx, ex):
            nonlocal total_chars
            if idx in used_indices:
                return False
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > _EXAMPLES_MAX_CHARS:
                return False
            parts.append(part)
            used_indices.add(idx)
            total_chars += len(part) + 2
            return True

        if confused_with:
            disambig_labels_needed = set(confused_with[:_TOP_CANDIDATE_LABELS])
            per_label_disambig = defaultdict(list)
            for score, idx, ex in ranked:
                if ex["target"] in disambig_labels_needed:
                    per_label_disambig[ex["target"]].append((score, idx, ex))
            for _round in range(_DISAMBIG_PER_CONFUSION):
                for lbl in confused_with[:_TOP_CANDIDATE_LABELS]:
                    pool = per_label_disambig.get(lbl, [])
                    for score, idx, ex in pool:
                        if idx not in used_indices:
                            try_add(idx, ex)
                            break
        for _, idx, ex in ranked:
            if total_chars >= _EXAMPLES_MAX_CHARS:
                break
            try_add(idx, ex)
        return parts

    def predict(self, input):
        parts = self._build_parts(input)
        examples_section = "\n\n".join(parts)
        label_section = self._label_section()
        prompt = PROMPT_TEMPLATE.format(
            label_section=label_section,
            examples_section=examples_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_selected": len(parts),
            "num_labels_in_vocab": len(self.label_vocab),
        }

    def learn_from_batch(self, batch_results):
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            ex = {"input": r["input"], "target": r["ground_truth"], "tokens": _tokenize(raw_q)}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)
            for lbl in r["ground_truth"].split(";"):
                lbl = lbl.strip()
                if lbl:
                    self.label_vocab[lbl] += 1
            if not r.get("was_correct", True):
                pred = r.get("prediction", "")
                gt = r["ground_truth"]
                if pred and pred != gt:
                    self.confusion[pred][gt] += 1

    def get_state(self):
        serialisable = [{k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()} for ex in self.examples]
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps({"examples": serialisable, "confusion": confusion_plain, "label_vocab": dict(self.label_vocab)}, indent=2)

    def set_state(self, state):
        data = json.loads(state)
        self.examples = []
        for ex in data.get("examples", []):
            restored = dict(ex)
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                raw_q = restored.get("raw_question", restored.get("input", ""))
                restored["tokens"] = _tokenize(raw_q)
            self.examples.append(restored)
        self.confusion = defaultdict(lambda: defaultdict(int))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = cnt
        self.label_vocab = defaultdict(int)
        for lbl, cnt in data.get("label_vocab", {}).items():
            self.label_vocab[lbl] = cnt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
BATCH = [
    {"input": "事实:...打砸财物，造成3166元损失...", "ground_truth": "故意毁坏财物",
     "prediction": "故意毁坏财物罪", "was_correct": False},
    {"input": "事实:...拐骗儿童...", "ground_truth": "拐骗儿童",
     "prediction": "拐骗儿童", "was_correct": True},
    {"input": "事实:...拐卖缅甸籍妇女...", "ground_truth": "拐卖妇女、儿童",
     "prediction": "拐卖妇女", "was_correct": False},
    {"input": "事实:...受贿非国家工作人员...", "ground_truth": "非国家工作人员受贿",
     "prediction": "受贿", "was_correct": False},
    {"input": "事实:...贩毒走私...", "ground_truth": "走私、贩卖、运输、制造毒品;徇私枉法",
     "prediction": "贩卖毒品;徇私枉法", "was_correct": False},
]

def fake_llm(prompt):
    # returns a label that's in the vocab
    return '{"reasoning": "test", "final_answer": "故意毁坏财物"}'

mem = LabelVocabMemory(fake_llm)

# cold-start: label section should be empty
ans, meta = mem.predict("事实:...测试...")
assert meta["num_labels_in_vocab"] == 0
assert meta["num_examples"] == 0
print("Cold start OK")

# learn
mem.learn_from_batch(BATCH)
assert "故意毁坏财物" in mem.label_vocab
assert "故意毁坏财物罪" not in mem.label_vocab, "wrong pred must not enter vocab"
assert "拐卖妇女、儿童" in mem.label_vocab
assert "走私、贩卖、运输、制造毒品" in mem.label_vocab
assert "徇私枉法" in mem.label_vocab
print(f"Vocab has {len(mem.label_vocab)} labels: {sorted(mem.label_vocab.keys())}")

# predict with vocab
ans, meta = mem.predict("事实:...破坏电力设备...")
assert meta["num_labels_in_vocab"] == 6  # 5 entries, last splits into 2 labels
label_sec = mem._label_section()
assert "故意毁坏财物" in label_sec
assert "拐卖妇女、儿童" in label_sec
print("Label section present in prompt after learning")

# state serialization round-trip
state = mem.get_state()
mem2 = LabelVocabMemory(fake_llm)
mem2.set_state(state)
assert len(mem2.examples) == len(mem.examples)
assert len(mem2.label_vocab) == len(mem.label_vocab)
assert mem2.label_vocab["故意毁坏财物"] == mem.label_vocab["故意毁坏财物"]
print("State round-trip OK")

# confusion matrix built correctly
assert "故意毁坏财物罪" in mem.confusion
assert mem.confusion["故意毁坏财物罪"]["故意毁坏财物"] == 1
print("Confusion matrix OK")

print("\nAll label_vocab_memory validation tests PASSED")
