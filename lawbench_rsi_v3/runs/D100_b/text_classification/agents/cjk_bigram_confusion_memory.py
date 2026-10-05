"""CJK Bigram Confusion Disambiguation Memory.

The base confusion_disambiguation_memory system uses a tokenizer that matches
only ASCII characters ([A-Za-z0-9]+).  For Chinese-language legal text this
produces near-zero overlap on every pair of examples — similarity is driven
solely by shared date/number strings, making retrieval essentially random and
making the confusion matrix signal useless (because every retrieved example
looks equally similar regardless of content).

This system replaces the tokenizer with CJK character bigrams plus ASCII
tokens.  Bigrams over Chinese characters give genuine semantic overlap that
tracks shared legal concepts and factual context, restoring the intended
behaviour of the confusion disambiguation retrieval algorithm.

Everything else — the confusion matrix, the disambiguation injection logic,
the prompt template, the serialisation — is identical to the base system.
The hypothesis is therefore tightly controlled: if accuracy improves, it is
because real similarity signal was absent and bigrams supply it.
"""

import json
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
_DISAMBIG_PER_CONFUSION = 2


def _tokenize(text: str) -> frozenset:
    """CJK character bigrams plus ASCII tokens.

    Bigrams capture shared legal concepts (e.g. '故意', '伤害', '故意伤害')
    while remaining language-agnostic — the same function works for any
    script where bigrams are meaningful units.
    """
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


class CjkBigramConfusionMemory(MemorySystem):
    """Confusion disambiguation with CJK-bigram similarity for Chinese text."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _scored(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize(query)
        result = [
            (_jaccard(q_tok, ex['tokens']), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _confusion_targets_for(self, labels: list[str]) -> list[str]:
        label_set = set(labels)
        counts: dict[str, int] = defaultdict(int)
        for lbl in labels:
            row = self.confusion.get(lbl, {})
            for actual, cnt in row.items():
                if actual not in label_set:
                    counts[actual] += cnt
        return [lbl for lbl, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        ranked = self._scored(query)

        top_labels = [ex['target'] for _, _, ex in ranked[:_TOP_CANDIDATE_LABELS]]
        confused_with = self._confusion_targets_for(top_labels)

        parts: list[str] = []
        used_indices: set[int] = set()
        total_chars = 0

        def try_add(idx: int, ex: dict) -> bool:
            nonlocal total_chars
            if idx in used_indices:
                return False
            q = ex.get('raw_question', ex['input'])
            part = f"Q: {q}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > MAX_CHARS:
                return False
            parts.append(part)
            used_indices.add(idx)
            total_chars += len(part) + 2
            return True

        if confused_with:
            disambig_labels_needed = set(confused_with[:_TOP_CANDIDATE_LABELS])
            per_label_disambig: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
            for score, idx, ex in ranked:
                if ex['target'] in disambig_labels_needed:
                    per_label_disambig[ex['target']].append((score, idx, ex))

            for _round in range(_DISAMBIG_PER_CONFUSION):
                for lbl in confused_with[:_TOP_CANDIDATE_LABELS]:
                    pool = per_label_disambig.get(lbl, [])
                    for score, idx, ex in pool:
                        if idx not in used_indices:
                            try_add(idx, ex)
                            break

        for _, idx, ex in ranked:
            if total_chars >= MAX_CHARS:
                break
            try_add(idx, ex)

        return parts

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = self._build_parts(input)
        examples_section = '\n\n'.join(parts)
        prompt = PROMPT_TEMPLATE.format(
            examples_section=examples_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, 'final_answer')
        return answer, {
            'full_response': response,
            'num_examples': len(self.examples),
            'num_selected': len(parts),
            'num_confusion_pairs': sum(len(v) for v in self.confusion.values()),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw_q = r.get('raw_question', r['input'])
            ex: dict[str, Any] = {
                'input': r['input'],
                'target': r['ground_truth'],
                'tokens': _tokenize(raw_q),
            }
            if 'raw_question' in r:
                ex['raw_question'] = r['raw_question']
            self.examples.append(ex)

            if not r.get('was_correct', True):
                pred = r.get('prediction', '')
                gt = r['ground_truth']
                if pred and pred != gt:
                    self.confusion[pred][gt] += 1

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps({
            'examples': serialisable,
            'confusion': confusion_plain,
        }, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        for ex in data.get('examples', []):
            restored = dict(ex)
            raw_q = restored.get('raw_question', restored.get('input', ''))
            restored['tokens'] = _tokenize(raw_q)
            self.examples.append(restored)

        self.confusion = defaultdict(lambda: defaultdict(int))
        for pred, actuals in data.get('confusion', {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = cnt
