"""Contrastive Boundary Memory.

Extends adaptive_tokenizer_confusion_memory with a structurally different prompt
architecture: rather than silently interleaving disambiguation examples into the
flat list, this system renders them in a dedicated "Common Confusions" section
that explicitly annotates each boundary example with the label NOT to predict.

Mechanism:
1. Same adaptive tokenizer + confusion-matrix retrieval as the frontier system.
2. At prompt construction time, examples selected for disambiguation are placed
   in a separate header section with a per-example "NOT: <wrong_label>" warning.
3. Regular similarity-ranked examples follow in a plain "Examples" section.

The hypothesis is that this explicit negative-example signal at the boundary gives
the model a targeted "avoid this mistake" cue, rather than forcing it to infer the
boundary from two similar positive examples — which the frontier system currently
relies on.
"""

import json
import re
from collections import defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# Two-section prompt: disambiguation warnings first, then regular examples.
# The disambiguation section annotates each example with the label to avoid,
# giving the model an explicit negative-example signal at the boundary.
_PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{disambig_section}{examples_section}

**Problem:**
{input}

**Instructions:**
- If disambiguation examples are shown above, pay close attention — they highlight inputs that look similar but have different correct answers
- Follow the patterns shown in the examples
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

_DISAMBIG_HEADER = "**Common Confusions — similar inputs with different answers:**\n\n"

MAX_CHARS = 30000
_TOP_CANDIDATE_LABELS = 3
_DISAMBIG_PER_CONFUSION = 2

# CJK Unicode ranges: CJK Unified, Hiragana, Katakana, Fullwidth/Halfwidth
_CJK_RE = re.compile(r'[一-鿿぀-ゟ゠-ヿ＀-￯]')


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _tokenize_word(text: str) -> frozenset:
    return frozenset(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _tokenize_bigram(text: str) -> frozenset:
    t = re.sub(r'\s+', ' ', text.strip())
    if len(t) < 2:
        return frozenset([t]) if t else frozenset()
    return frozenset(t[i:i + 2] for i in range(len(t) - 1))


def _tokenize_adaptive(text: str) -> frozenset:
    """Route to bigrams for CJK text, word tokens for ASCII/Latin."""
    return _tokenize_bigram(text) if _has_cjk(text) else _tokenize_word(text)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class ContrastiveBoundaryMemory(MemorySystem):
    """Confusion-matrix retrieval with explicit contrastive boundary prompting.

    Disambiguation examples are placed in a dedicated header section with a
    per-example 'NOT: <wrong_label>' annotation, giving the model an explicit
    negative-example signal rather than burying boundary examples in the flat list.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        # confusion[predicted][actual] = count — tracks what the model got wrong
        self.confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _scored(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _confusion_targets_for(self, labels: list[str]) -> list[str]:
        """Return labels that were historically confused with any of the given labels."""
        label_set = set(labels)
        counts: dict[str, int] = defaultdict(int)
        for lbl in labels:
            row = self.confusion.get(lbl, {})
            for actual, cnt in row.items():
                if actual not in label_set:
                    counts[actual] += cnt
        return [lbl for lbl, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    def _build_prompt(self, query: str) -> tuple[str, int, int]:
        """Return (prompt_text, num_disambig, num_regular)."""
        if not self.examples:
            return _PROMPT_TEMPLATE.format(
                disambig_section="",
                examples_section="",
                input=query,
            ), 0, 0

        ranked = self._scored(query)
        top_labels = [ex["target"] for _, _, ex in ranked[:_TOP_CANDIDATE_LABELS]]
        confused_with = self._confusion_targets_for(top_labels)

        # wrong_label_for[confused_target] = the predicted label that was wrong
        # Used to annotate each disambiguation example with "NOT: <predicted_label>"
        wrong_label_for: dict[str, str] = {}
        if confused_with:
            for pred_lbl in top_labels:
                for actual, cnt in self.confusion.get(pred_lbl, {}).items():
                    if actual not in set(top_labels) and actual not in wrong_label_for:
                        wrong_label_for[actual] = pred_lbl

        disambig_parts: list[str] = []
        regular_parts: list[str] = []
        used_indices: set[int] = set()
        total_chars = 0

        def try_add_disambig(idx: int, ex: dict, wrong_label: str) -> bool:
            nonlocal total_chars
            if idx in used_indices:
                return False
            q = ex.get("raw_question", ex["input"])
            # Explicit negative annotation: tells model which wrong answer to avoid
            part = f"Q: {q}\nA: {ex['target']}  [NOT: {wrong_label}]"
            if total_chars + len(part) + 2 > MAX_CHARS:
                return False
            disambig_parts.append(part)
            used_indices.add(idx)
            total_chars += len(part) + 2
            return True

        def try_add_regular(idx: int, ex: dict) -> bool:
            nonlocal total_chars
            if idx in used_indices:
                return False
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > MAX_CHARS:
                return False
            regular_parts.append(part)
            used_indices.add(idx)
            total_chars += len(part) + 2
            return True

        # Phase 1: populate disambiguation section
        if confused_with:
            disambig_labels_needed = set(confused_with[:_TOP_CANDIDATE_LABELS])
            per_label: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
            for score, idx, ex in ranked:
                if ex["target"] in disambig_labels_needed:
                    per_label[ex["target"]].append((score, idx, ex))

            for _round in range(_DISAMBIG_PER_CONFUSION):
                for lbl in confused_with[:_TOP_CANDIDATE_LABELS]:
                    pool = per_label.get(lbl, [])
                    wrong = wrong_label_for.get(lbl, top_labels[0] if top_labels else "")
                    for score, idx, ex in pool:
                        if idx not in used_indices:
                            try_add_disambig(idx, ex, wrong)
                            break

        # Phase 2: fill remaining budget with similarity-ranked regular examples
        for _, idx, ex in ranked:
            if total_chars >= MAX_CHARS:
                break
            try_add_regular(idx, ex)

        # Build prompt sections
        if disambig_parts:
            disambig_section = _DISAMBIG_HEADER + "\n\n".join(disambig_parts) + "\n\n"
        else:
            disambig_section = ""

        if regular_parts:
            examples_section = "**Examples:**\n\n" + "\n\n".join(regular_parts)
        else:
            examples_section = ""

        prompt = _PROMPT_TEMPLATE.format(
            disambig_section=disambig_section,
            examples_section=examples_section,
            input=query,
        )
        return prompt, len(disambig_parts), len(regular_parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        prompt, num_disambig, num_regular = self._build_prompt(input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_disambig": num_disambig,
            "num_regular": num_regular,
            "num_confusion_pairs": sum(len(v) for v in self.confusion.values()),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            ex: dict[str, Any] = {
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": _tokenize_adaptive(raw_q),
            }
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

            if not r.get("was_correct", True):
                pred = r.get("prediction", "")
                gt = r["ground_truth"]
                if pred and pred != gt:
                    self.confusion[pred][gt] += 1

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps({"examples": serialisable, "confusion": confusion_plain}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        for ex in data.get("examples", []):
            restored = dict(ex)
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                raw_q = restored.get("raw_question", restored.get("input", ""))
                restored["tokens"] = _tokenize_adaptive(raw_q)
            self.examples.append(restored)
        self.confusion = defaultdict(lambda: defaultdict(int))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = cnt
