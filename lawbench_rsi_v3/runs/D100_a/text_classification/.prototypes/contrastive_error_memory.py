"""Validation prototype for contrastive_error_memory.py — tests imports and mechanism."""

import json
import re
from collections import defaultdict
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
# Inline implementation (same logic as agents/contrastive_error_memory.py)
# ---------------------------------------------------------------------------
_EXAMPLES_MAX_CHARS = 26000
_MISTAKES_MAX_CHARS = 4000
_TOP_CANDIDATE_LABELS = 3
_DISAMBIG_PER_CONFUSION = 2
_MAX_MISTAKE_PAIRS = 5

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{mistakes_section}{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Avoid the common mistakes listed above
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


class ContrastiveErrorMemory(MemorySystem):
    def __init__(self, llm):
        super().__init__(llm)
        self.examples = []
        self.confusion = defaultdict(lambda: defaultdict(int))
        self.error_pairs = []

    def _top_error_pairs(self, query):
        if not self.error_pairs:
            return []
        q_tok = _tokenize(query)
        scored = sorted(self.error_pairs, key=lambda p: _jaccard(q_tok, p["tokens"]), reverse=True)
        return scored[:_MAX_MISTAKE_PAIRS]

    def _build_mistakes_section(self, query):
        pairs = self._top_error_pairs(query)
        if not pairs:
            return ""
        lines = ["## Common Mistakes (similar cases — avoid these errors):"]
        total = len(lines[0])
        for p in pairs:
            line = f"  ✗ Wrong: {p['wrong']}  →  ✓ Correct: {p['correct']}"
            if total + len(line) + 1 > _MISTAKES_MAX_CHARS:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines) + "\n\n"

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
        mistakes_section = self._build_mistakes_section(input)
        parts = self._build_parts(input)
        examples_section = "\n\n".join(parts)
        prompt = PROMPT_TEMPLATE.format(
            mistakes_section=mistakes_section,
            examples_section=examples_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_selected": len(parts),
            "num_error_pairs": len(self.error_pairs),
            "num_mistakes_shown": len(self._top_error_pairs(input)),
        }

    def learn_from_batch(self, batch_results):
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            ex = {"input": r["input"], "target": r["ground_truth"], "tokens": _tokenize(raw_q)}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)
            if not r.get("was_correct", True):
                pred = r.get("prediction", "")
                gt = r["ground_truth"]
                if pred and pred != gt:
                    self.confusion[pred][gt] += 1
                    self.error_pairs.append({"tokens": _tokenize(raw_q), "wrong": pred, "correct": gt})

    def get_state(self):
        serialisable = [{k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()} for ex in self.examples]
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        pairs_serialisable = [{k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in p.items()} for p in self.error_pairs]
        return json.dumps({"examples": serialisable, "confusion": confusion_plain, "error_pairs": pairs_serialisable}, indent=2)

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
        self.error_pairs = []
        for p in data.get("error_pairs", []):
            restored = dict(p)
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                restored["tokens"] = frozenset()
            self.error_pairs.append(restored)


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
    {"input": "事实:...挪用公款用于个人经营...", "ground_truth": "挪用资金",
     "prediction": "挪用公款", "was_correct": False},
    {"input": "事实:...贩毒走私...", "ground_truth": "走私、贩卖、运输、制造毒品;徇私枉法",
     "prediction": "贩卖毒品;徇私枉法", "was_correct": False},
]

def fake_llm(prompt):
    # verify mistakes section is in prompt when error pairs exist
    if "Common Mistakes" in prompt:
        return '{"reasoning": "found mistakes section", "final_answer": "故意毁坏财物"}'
    return '{"reasoning": "no mistakes yet", "final_answer": "盗窃"}'

mem = ContrastiveErrorMemory(fake_llm)

# cold start — no error pairs, no mistakes section
ans, meta = mem.predict("事实:...测试...")
assert meta["num_error_pairs"] == 0
assert meta["num_mistakes_shown"] == 0
print("Cold start OK — no mistakes section")

# learn batch
mem.learn_from_batch(BATCH)
assert len(mem.error_pairs) == 5  # 5 wrong predictions
assert len(mem.examples) == 6
print(f"Learned {len(mem.error_pairs)} error pairs, {len(mem.examples)} examples")

# predict with mistakes
query = "事实:...挪用单位资金，用于个人股票投资..."
ans, meta = mem.predict(query)
assert meta["num_mistakes_shown"] > 0, "should show mistakes for similar query"
print(f"Mistakes shown for '挪用' query: {meta['num_mistakes_shown']}")

# verify the top mistake is the 挪用 pair
top_pairs = mem._top_error_pairs(query)
assert top_pairs[0]["wrong"] == "挪用公款", f"Expected 挪用公款 first, got {top_pairs[0]['wrong']}"
assert top_pairs[0]["correct"] == "挪用资金"
print(f"Top mistake for '挪用' query: {top_pairs[0]['wrong']} → {top_pairs[0]['correct']}")

# verify mistakes section appears in prompt
mistakes_sec = mem._build_mistakes_section(query)
assert "Common Mistakes" in mistakes_sec
assert "挪用公款" in mistakes_sec
assert "挪用资金" in mistakes_sec
print("Mistakes section content correct")

# confusion matrix
assert mem.confusion["挪用公款"]["挪用资金"] == 1
assert mem.confusion["拐卖妇女"]["拐卖妇女、儿童"] == 1
print("Confusion matrix OK")

# state round-trip
state = mem.get_state()
mem2 = ContrastiveErrorMemory(fake_llm)
mem2.set_state(state)
assert len(mem2.error_pairs) == len(mem.error_pairs)
assert len(mem2.examples) == len(mem.examples)
assert mem2.error_pairs[0]["wrong"] == mem.error_pairs[0]["wrong"]
assert mem2.error_pairs[0]["correct"] == mem.error_pairs[0]["correct"]
assert isinstance(mem2.error_pairs[0]["tokens"], frozenset)
print("State round-trip OK")

print("\nAll contrastive_error_memory validation tests PASSED")
