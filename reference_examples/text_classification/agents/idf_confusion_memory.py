"""IDF-weighted Confusion Disambiguation Memory.

Extends fixed_confusion_memory with a fundamentally different similarity
scoring mechanism: IDF-weighted Jaccard, where each token's contribution is
scaled by its inverse document frequency across stored examples.

Problem with plain Jaccard: every stored example shares a large pool of
common tokens (prepositions, punctuation bigrams, short chemistry fragments
like 'c', 'o', 'n', common CJK bigrams in boilerplate preambles).  These
dominate intersection counts and make retrieval nearly independent of the
actually discriminative tokens — a specific functional group in a SMILES
string, a rare legal term, a domain-specific keyword.

IDF rebalances this: common tokens contribute less to the similarity score
and rare tokens contribute more.  The formula for weighted Jaccard is:

    score = Σ idf(t) for t in (A ∩ B)
            ─────────────────────────────
            Σ idf(t) for t in (A ∪ B)

IDF is computed lazily and updated in bulk after each learning step from the
current example pool, so it tracks the actual vocabulary distribution.

The normalization fix from fixed_confusion_memory is also included:
predictions are stripped of format wrappers before being stored as confusion
keys.

Retrieval algorithm:
1. Compute IDF over all stored example token sets.
2. Score all stored examples with IDF-weighted Jaccard against the query.
3. Identify confusion targets for the top-3 candidate labels (normalised).
4. Inject one disambiguation example per confused label (round-robin, 2 rounds).
5. Fill remaining budget with IDF-similarity-ranked examples.
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
_DISAMBIG_PER_CONFUSION = 2

_CJK_RE = re.compile(r'[一-鿿぀-ゟ゠-ヿ＀-￯]')


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _tokenize_word(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _tokenize_bigram(text: str) -> list[str]:
    t = re.sub(r'\s+', ' ', text.strip())
    if len(t) < 2:
        return [t] if t else []
    return [t[i:i + 2] for i in range(len(t) - 1)]


def _tokenize_adaptive(text: str) -> list[str]:
    return _tokenize_bigram(text) if _has_cjk(text) else _tokenize_word(text)


def _compute_idf(token_sets: list[frozenset]) -> dict[str, float]:
    """Compute IDF over a list of token frozensets.

    Uses the standard smoothed formula:  idf(t) = log((N+1)/(df+1)) + 1
    so that every token receives a positive weight even when it appears in
    every document.
    """
    N = len(token_sets)
    df: dict[str, int] = defaultdict(int)
    for ts in token_sets:
        for t in ts:
            df[t] += 1
    return {t: math.log((N + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}


def _idf_jaccard(a_tokens: frozenset, b_tokens: frozenset, idf: dict[str, float]) -> float:
    """Weighted Jaccard using IDF weights; falls back to 0 for empty sets."""
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    inter = a_tokens & b_tokens
    w_inter = sum(idf.get(t, 1.0) for t in inter)
    w_union = sum(idf.get(t, 1.0) for t in union)
    if w_union == 0.0:
        return 0.0
    return w_inter / w_union


def _normalize_prediction(pred: str) -> str:
    """Strip format wrappers so confusion keys align with ground-truth labels."""
    pred = re.sub(r'^\[[^\]]*\]\s*', '', pred)
    pred = re.sub(r'\s*<[^>]+>\s*$', '', pred)
    return pred.strip()


class IDFConfusionMemory(MemorySystem):
    """IDF-weighted retrieval with normalised confusion-matrix disambiguation."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Cached IDF; rebuilt whenever examples change
        self._idf: dict[str, float] = {}
        self._idf_dirty: bool = False

    def _get_idf(self) -> dict[str, float]:
        if self._idf_dirty or not self._idf:
            token_sets = [ex["tokens"] for ex in self.examples]
            self._idf = _compute_idf(token_sets) if token_sets else {}
            self._idf_dirty = False
        return self._idf

    def _scored(self, query: str) -> list[tuple[float, int, dict]]:
        idf = self._get_idf()
        q_tok = frozenset(_tokenize_adaptive(query))
        result = [
            (_idf_jaccard(q_tok, ex["tokens"], idf), idx, ex)
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

        if confused_with:
            disambig_labels_needed = set(confused_with[:_TOP_CANDIDATE_LABELS])
            per_label_disambig: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
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
            "num_confusion_pairs": sum(len(v) for v in self.confusion.values()),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw_q = r.get("raw_question", r["input"])
            ex: dict[str, Any] = {
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": frozenset(_tokenize_adaptive(raw_q)),
            }
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

            if not r.get("was_correct", True):
                pred = r.get("prediction", "")
                gt = r["ground_truth"]
                if pred and pred != gt:
                    pred_clean = _normalize_prediction(pred)
                    if pred_clean and pred_clean != gt:
                        self.confusion[pred_clean][gt] += 1

        # Invalidate IDF cache after any batch update
        self._idf_dirty = True

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
                restored["tokens"] = frozenset(_tokenize_adaptive(raw_q))
            self.examples.append(restored)
        self.confusion = defaultdict(lambda: defaultdict(int))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = cnt
        self._idf_dirty = True
