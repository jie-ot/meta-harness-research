"""Discriminative-Similarity Confusion Memory.

Standard Jaccard treats every token in the overlap equally. A token that appears
in examples spanning many different labels is essentially a stop-token for the
retrieval task: matching it tells you little about which label the query belongs
to. A token that appears in examples of only one or two labels is highly
discriminative and should contribute much more to the similarity score.

This system replaces flat Jaccard with a label-discriminativeness-weighted
similarity:

    sim(q, e) = sum(w(t) for t in q∩e) / sum(w(t) for t in q∪e)

where w(t) = 1 / label_diversity(t), and label_diversity(t) is the number of
distinct labels among all stored examples that contain token t. Tokens never seen
during training get weight 1.0 (maximally discriminative — we have no evidence
they are common).

The confusion disambiguation phase (identical to the frontier) is preserved,
since it is independently proven to help.

Why this differs from idf_confusion_memory (iter 8):
- idf_confusion used standard IDF = log(N / df_t), which weights by rarity in
  the document corpus. A token can be rare yet appear in many different labels.
- This system weights by label diversity directly: a common token that stays
  exclusive to one label (e.g. a disease-specific symptom) retains high weight,
  while a common token spread across many labels (e.g. legal boilerplate) is
  down-weighted regardless of its document frequency.
"""

import json
import math
import re
from collections import defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000
_TOP_CANDIDATE_LABELS = 3
_DISAMBIG_SLOTS = 3

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
    return _tokenize_bigram(text) if _has_cjk(text) else _tokenize_word(text)


class DiscriminativeSimilarityMemory(MemorySystem):
    """Confusion disambiguation with label-discriminativeness-weighted similarity.

    Retrieval scoring replaces flat Jaccard with discriminative-weighted Jaccard:
    tokens shared by few labels get high weight; tokens common across all labels
    contribute little. Everything else — pool management, confusion phase, prompt
    format — matches the frontier.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        # token -> set of labels that have at least one example containing that token
        self.token_labels: dict[str, set[str]] = defaultdict(set)
        # confusion[predicted_label][actual_label] = count
        self.confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _discriminative_weight(self, token: str) -> float:
        """1 / number of distinct labels containing this token; 1.0 if unseen."""
        n = len(self.token_labels.get(token, ()))
        return 1.0 if n == 0 else 1.0 / n

    def _sim(self, q_tok: frozenset, ex_tok: frozenset) -> float:
        """Discriminativeness-weighted Jaccard similarity."""
        union = q_tok | ex_tok
        if not union:
            return 0.0
        intersection = q_tok & ex_tok
        num = sum(self._discriminative_weight(t) for t in intersection)
        den = sum(self._discriminative_weight(t) for t in union)
        return num / den if den > 0 else 0.0

    def _scored(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = [
            (self._sim(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _confusion_targets_for(self, labels: list[str]) -> list[str]:
        label_set = set(labels)
        counts: dict[str, int] = defaultdict(int)
        for lbl in labels:
            for actual, cnt in self.confusion.get(lbl, {}).items():
                if actual not in label_set:
                    counts[actual] += cnt
        return [lbl for lbl, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        ranked = self._scored(query)
        top_labels = [ex["target"] for _, _, ex in ranked[:_TOP_CANDIDATE_LABELS]]
        confused_with = self._confusion_targets_for(top_labels)

        parts: list[str] = []
        used_indices: set[int] = set()
        total_chars = 0

        def try_add(idx: int, ex: dict) -> bool:
            nonlocal total_chars
            if idx in used_indices:
                return False
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > MAX_CHARS:
                return False
            parts.append(part)
            used_indices.add(idx)
            total_chars += len(part) + 2
            return True

        # Phase 1: disambiguation — best sim-ranked example per confusion target
        if confused_with:
            per_label: dict[str, int | None] = {
                lbl: None for lbl in confused_with[:_DISAMBIG_SLOTS]
            }
            disambig_set = set(per_label.keys())
            for _, idx, ex in ranked:
                if ex["target"] in disambig_set and per_label[ex["target"]] is None:
                    per_label[ex["target"]] = idx
                if all(v is not None for v in per_label.values()):
                    break
            for lbl in confused_with[:_DISAMBIG_SLOTS]:
                if per_label.get(lbl) is not None:
                    try_add(per_label[lbl], self.examples[per_label[lbl]])

        # Phase 2: discriminative-similarity fill
        for _, idx, ex in ranked:
            if total_chars >= MAX_CHARS:
                break
            try_add(idx, ex)

        return parts

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = self._build_parts(input)
        examples_section = "\n\n".join(parts)
        prompt = PROMPT_TEMPLATE.format(
            examples_section=examples_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_selected": len(parts),
            "num_token_types": len(self.token_labels),
            "num_confusion_pairs": sum(len(v) for v in self.confusion.values()),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            tokens = _tokenize_adaptive(raw_q)
            label = r["ground_truth"]
            ex: dict[str, Any] = {
                "input": r["input"],
                "target": label,
                "tokens": tokens,
            }
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

            # Update token→label index
            for tok in tokens:
                self.token_labels[tok].add(label)

            # Update confusion matrix
            if not r.get("was_correct", True):
                pred = r.get("prediction", "")
                if pred and pred != label:
                    self.confusion[pred][label] += 1

    def get_state(self) -> str:
        def serialise_ex(ex: dict) -> dict:
            return {
                k: (sorted(v) if isinstance(v, frozenset) else v)
                for k, v in ex.items()
            }

        token_labels_ser = {tok: sorted(lbls) for tok, lbls in self.token_labels.items()}
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps(
            {
                "examples": [serialise_ex(ex) for ex in self.examples],
                "token_labels": token_labels_ser,
                "confusion": confusion_plain,
            },
            indent=2,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)

        def restore_ex(ex: dict) -> dict:
            restored = dict(ex)
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                raw_q = restored.get("raw_question", restored.get("input", ""))
                restored["tokens"] = _tokenize_adaptive(raw_q)
            return restored

        self.examples = [restore_ex(ex) for ex in data.get("examples", [])]
        self.token_labels = defaultdict(set)
        for tok, lbls in data.get("token_labels", {}).items():
            self.token_labels[tok] = set(lbls)
        self.confusion = defaultdict(lambda: defaultdict(int))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = cnt
