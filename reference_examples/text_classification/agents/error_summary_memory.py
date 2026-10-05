"""Error Summary Memory.

Extends decayed_confusion_memory with a learning-time LLM synthesis (axis F):
after each batch containing enough errors, calls the LLM once to generate a
concise 2-3 sentence "Current weaknesses" narrative from recent wrong predictions.
That narrative is injected above the few-shot examples in the predict prompt.

This is distinct from all prior LLM-in-learning approaches:
- reflexion_memory (iter 1): replaced examples entirely with LLM lessons — no examples
- llm_rule_synthesis_memory (iter 7): generated O(N^2) pairwise rules, too expensive
- recent_error_hint_memory (iter 14): showed raw wrong→correct pairs (no synthesis)

This approach generates one O(1) paragraph per batch that sits alongside the
existing example retrieval. The hint captures the *pattern* of errors in natural
language ("tends to predict X when the answer is Y because…") rather than
enumerating individual cases. The hint is updated periodically — only when
there are enough new errors to form a meaningful pattern — keeping LLM cost low.
"""

import json
import re
from collections import defaultdict, deque
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# Predict prompt: hint section is optional (omitted when no hint yet)
PROMPT_TEMPLATE_WITH_HINT = """Solve the problem below based on the examples provided.

**Current weaknesses to watch for:**
{hint}

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

PROMPT_TEMPLATE_NO_HINT = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

SYNTHESIS_PROMPT = """A classifier has made the following errors recently (predicted answer vs. correct answer):

{error_lines}

Write 2-3 sentences describing the main mistake patterns and how to avoid them.
Be concise and actionable. Focus on general reasoning strategies, not on specific answers.
Do not list individual errors — synthesize the pattern.

{{"hint": "[your 2-3 sentence hint]"}}"""

MAX_CHARS = 30000
HINT_BUDGET = 400           # chars reserved for hint in context
_TOP_CANDIDATE_LABELS = 3
_DISAMBIG_PER_CONFUSION = 2

DECAY_FACTOR = 0.95
PRUNE_THRESHOLD = 0.05

MIN_ERRORS_FOR_SYNTHESIS = 5    # minimum errors in window to trigger LLM synthesis
ERROR_WINDOW_SIZE = 15          # rolling window of recent errors used for synthesis
MAX_ERRORS_IN_PROMPT = 10       # errors included in synthesis prompt

_CJK_RE = re.compile(r'[一-鿿぀-ゟ゠-ヿ＀-￯]')
_WRAPPER_RE = re.compile(r'^\s*\[[^\]]*\]|<eoa>\s*$|\[[^\]]*\]\s*$', re.IGNORECASE)


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


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_prediction(pred: str) -> str:
    """Strip format wrappers so predictions match ground-truth label strings."""
    if not pred:
        return pred
    cleaned = _WRAPPER_RE.sub('', pred).strip()
    return cleaned if cleaned else pred


class ErrorSummaryMemory(MemorySystem):
    """Decayed confusion memory with periodic LLM-synthesized error hint injection."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.confusion: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Rolling window of recent (pred, gt) error pairs for synthesis
        self.recent_errors: deque[tuple[str, str]] = deque(maxlen=ERROR_WINDOW_SIZE)
        self.current_hint: str = ""
        # Track how many errors were in window when hint was last synthesized
        self._hint_error_count: int = 0

    def _scored(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _confusion_targets_for(self, labels: list[str]) -> list[str]:
        label_set = set(labels)
        counts: dict[str, float] = defaultdict(float)
        for lbl in labels:
            row = self.confusion.get(lbl, {})
            for actual, cnt in row.items():
                if actual not in label_set:
                    counts[actual] += cnt
        return [lbl for lbl, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    def _build_parts(self, query: str) -> list[str]:
        if not self.examples:
            return []

        # Reduce budget if a hint will be shown
        effective_budget = MAX_CHARS - HINT_BUDGET if self.current_hint else MAX_CHARS

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
            if total_chars + len(part) + 2 > effective_budget:
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
            if total_chars >= effective_budget:
                break
            try_add(idx, ex)

        return parts

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = self._build_parts(input)
        examples_section = "\n\n".join(parts)

        if self.current_hint:
            prompt = PROMPT_TEMPLATE_WITH_HINT.format(
                hint=self.current_hint,
                examples_section=examples_section,
                input=input,
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_HINT.format(
                examples_section=examples_section,
                input=input,
            )

        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_selected": len(parts),
            "has_hint": bool(self.current_hint),
        }

    def _maybe_synthesize_hint(self) -> None:
        """Call LLM to synthesize a new hint if there are enough fresh errors."""
        n_errors = len(self.recent_errors)
        if n_errors < MIN_ERRORS_FOR_SYNTHESIS:
            return
        # Only re-synthesize when we have meaningfully more errors than last time
        if n_errors <= self._hint_error_count:
            return

        errors_for_prompt = list(self.recent_errors)[-MAX_ERRORS_IN_PROMPT:]
        error_lines = "\n".join(
            f"- Predicted: {pred[:80]} | Correct: {gt[:80]}"
            for pred, gt in errors_for_prompt
        )
        synthesis_prompt = SYNTHESIS_PROMPT.format(error_lines=error_lines)
        try:
            response = self.call_llm(synthesis_prompt)
            hint = extract_json_field(response, "hint")
            if hint and hint != response and len(hint) > 10:
                self.current_hint = hint[:HINT_BUDGET]
                self._hint_error_count = n_errors
        except Exception:
            pass  # keep existing hint on failure

    def _decay_confusion(self) -> None:
        to_delete_outer = []
        for pred, actuals in self.confusion.items():
            to_delete_inner = []
            for actual in actuals:
                actuals[actual] *= DECAY_FACTOR
                if actuals[actual] < PRUNE_THRESHOLD:
                    to_delete_inner.append(actual)
            for actual in to_delete_inner:
                del actuals[actual]
            if not actuals:
                to_delete_outer.append(pred)
        for pred in to_delete_outer:
            del self.confusion[pred]

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
                pred = _normalize_prediction(r.get("prediction", ""))
                gt = r["ground_truth"]
                if pred and pred != gt:
                    self.confusion[pred][gt] += 1.0
                    self.recent_errors.append((pred, gt))

        self._decay_confusion()
        self._maybe_synthesize_hint()

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        confusion_plain = {k: dict(v) for k, v in self.confusion.items()}
        return json.dumps({
            "examples": serialisable,
            "confusion": confusion_plain,
            "recent_errors": list(self.recent_errors),
            "current_hint": self.current_hint,
            "hint_error_count": self._hint_error_count,
        }, indent=2)

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
        self.confusion = defaultdict(lambda: defaultdict(float))
        for pred, actuals in data.get("confusion", {}).items():
            for actual, cnt in actuals.items():
                self.confusion[pred][actual] = float(cnt)
        self.recent_errors = deque(
            [tuple(e) for e in data.get("recent_errors", [])],
            maxlen=ERROR_WINDOW_SIZE,
        )
        self.current_hint = data.get("current_hint", "")
        self._hint_error_count = data.get("hint_error_count", 0)
