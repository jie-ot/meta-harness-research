"""Canonical Charge Lexicon Memory — makes the model output a statutory *string*, not a concept.

Diagnosis this addresses (largest failure bucket): the model identifies the right
conduct but names it with a non-statutory paraphrase, and scoring is exact string
match, so it scores zero.
    model: 侵犯注册商标专用权罪   truth: 假冒注册商标
    model: 非法种植罂粟           truth: 非法种植毒品原植物
    model: 故意毁坏财物罪         truth: 故意毁坏财物
    model: 非法制造枪支           truth: 非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物

Mechanism (not a parameter change — an added stage the base system does not have):
the system maintains a *lexicon* of exact statutory strings, harvested from every
ground truth it is ever shown, and it rewrites its own answer into a lexicon member
before returning it. Two independently learned rule layers feed that rewrite:

  1. Verbatim payload rules — recorded whenever the model's answer differs from the
     truth. This is the load-bearing layer, because the failure is often a *verbatim*
     repeat with zero character overlap (开设赌场 vs 赌博); no similarity metric can
     recover those, only the stored observation can.
  2. Alignment rules — when the answer and truth have the same number of charges,
     each answer part is aligned to the nearest truth part and that rewrite is stored.
     Handles the paraphrase family (故意毁坏财物罪 -> 故意毁坏财物).

Resolution order at output time is deterministic: lexicon hit, verbatim rule,
alignment rule, trailing-"罪" strip, containment (statutory enumerations get
truncated by the model), then a conservative character-bigram fallback. The lexicon
also enters the *prompt* as an allowed vocabulary, so the model is asked to pick an
exact string up front rather than being corrected only after the fact.

Character bigrams (not `[A-Za-z0-9]+` tokens) are used throughout, because the base
tokenizer leaves only years and amounts on Chinese text — see the retrieval note in
the iteration-1 diagnosis.
"""

import json
import re
from collections import defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Label the statutory charge(s) for the fact below.

**Charge name lexicon — when a listed name applies, reproduce it verbatim:**
{lexicon_section}
{examples_section}
**Fact:**
{input}

**Instructions:**
- Name each charge by its exact statutory string as listed in the lexicon above.
- If several charges apply, join them with ';' on one line.
- Do not append or drop characters. A near-miss name scores as wrong.
- If no listed name fits, use the correct statutory name anyway.
- Put the charge(s) only inside [罪名] and <eoa>.

{{"reasoning": "[brief reasoning]", "final_answer": "[罪名]<charge(s)><eoa>"}}"""

# Terse retry used only when the first call returns nothing (long facts push the
# answer into the truncated reasoning channel).
RETRY_TEMPLATE = """Give the statutory charge(s) for this fact. Answer on one line.

{input}

Format: {{"final_answer": "[罪名]<charge(s)><eoa>"}}"""

MAX_LEXICON_ENTRIES = 60
MAX_EXAMPLES = 4
MAX_EXAMPLE_CHARS = 220
_FUZZY_THRESHOLD = 0.6
_STATE_VERSION = 1


def _clean(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _split_charges(s: str) -> list[str]:
    """Split on ';' only — '、' is internal to statutory names and must not split."""
    return [p.strip() for p in re.split(r"[;；]", s or "") if p.strip()]


def _payload(s: str) -> str:
    """Everything between [罪名] and <eoa> (or the whole string if unmarked)."""
    m = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", s or "", re.S)
    return (m.group(1) if m else (s or "")).strip()


def _wrap(payload: str) -> str:
    return f"[罪名]{payload}<eoa>" if payload else ""


def _bigrams(s: str) -> frozenset[str]:
    s = _clean(s)
    if not s:
        return frozenset()
    if len(s) == 1:
        return frozenset([s])
    return frozenset(s[i : i + 2] for i in range(len(s) - 1))


def _sim(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


class CanonicalChargeLexiconMemory(MemorySystem):
    """Lexicon-constrained charge naming with learned rewrite rules."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        # canonical statutory string -> times observed as a ground truth
        self.lexicon: dict[str, int] = {}
        # whole answer payload -> {truth payload: count}
        self.payload_rules: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # single answer part -> {truth part: count}
        self.part_rules: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # (fact, exact truth string) pairs, most recent first
        self.examples: list[dict[str, str]] = []
        self._learned_items = 0

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            truth = _clean(_payload(r.get("ground_truth", "")))
            if not truth:
                continue
            for t in _split_charges(truth):
                t = _clean(t)
                if t:
                    self.lexicon[t] = self.lexicon.get(t, 0) + 1

            pred = _clean(_payload(r.get("prediction", "")))
            if pred and pred != truth:
                # Layer 1: the literal rewrite the model needed. Survives zero
                # character overlap between the two strings.
                self.payload_rules[pred][truth] += 1
                # Layer 2: part-wise alignment when the two agree on charge count.
                pred_parts = [_clean(p) for p in _split_charges(pred)]
                truth_parts = [_clean(t) for t in _split_charges(truth)]
                if len(pred_parts) == len(truth_parts) and len(pred_parts) > 1:
                    for p in pred_parts:
                        best, best_s = None, 0.0
                        for t in truth_parts:
                            s = _sim(p, t)
                            if s > best_s:
                                best, best_s = t, s
                        if best and best != p:
                            self.part_rules[p][best] += 1

            raw_q = r.get("raw_question", r.get("input", ""))
            self.examples.append({"fact": raw_q, "target": truth})
            self._learned_items += 1

        # Keep the example buffer small: it is a style hint, not the memory.
        if len(self.examples) > MAX_EXAMPLES * 4:
            self.examples = self.examples[-MAX_EXAMPLES * 4 :]

    # ------------------------------------------------------------------
    # Output normalization — deterministic, no LLM
    # ------------------------------------------------------------------

    def _resolve_part(self, cand: str) -> str:
        c = _clean(cand)
        if not c:
            return ""
        if c in self.lexicon:
            return c
        if c in self.part_rules and self.part_rules[c]:
            return max(self.part_rules[c].items(), key=lambda kv: kv[1])[0]
        if c.endswith("罪") and c[:-1] in self.lexicon:
            return c[:-1]
        contained = [t for t in self.lexicon if c in t or t in c]
        if contained:
            # Statutory enumerations are the long form; prefer the longest match.
            return max(contained, key=len)
        best, best_s = c, _FUZZY_THRESHOLD
        for t in self.lexicon:
            s = _sim(c, t)
            if s > best_s:
                best, best_s = t, s
        return best

    def _canonicalize(self, raw: str) -> str:
        payload = _clean(_payload(raw))
        if not payload:
            return ""
        # Whole-payload verbatim rule first: it can map names with no shared
        # characters, which per-part similarity cannot.
        if payload in self.payload_rules and self.payload_rules[payload]:
            mapped = max(self.payload_rules[payload].items(), key=lambda kv: kv[1])[0]
            if mapped:
                return _wrap(mapped)
        out, seen = [], set()
        for p in _split_charges(payload):
            c = self._resolve_part(p)
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return _wrap(";".join(out))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _lexicon_section(self, query: str) -> str:
        if not self.lexicon:
            return "(empty — name the charge as the statute does)"
        entries = sorted(
            self.lexicon.items(),
            key=lambda kv: (-_sim(query, kv[0]), -kv[1], kv[0]),
        )[:MAX_LEXICON_ENTRIES]
        return "\n".join(f"- {name}" for name, _ in entries)

    def _examples_section(self, query: str) -> str:
        if not self.examples:
            return ""
        ranked = sorted(
            self.examples,
            key=lambda ex: -_sim(query, ex["fact"]),
        )[:MAX_EXAMPLES]
        blocks = []
        for ex in ranked:
            fact = _clean(ex["fact"])[:MAX_EXAMPLE_CHARS]
            blocks.append(f"Example fact: {fact}\nCharge: {ex['target']}")
        return "\n\n" + "\n\n".join(blocks) + "\n"

    # ------------------------------------------------------------------
    # MemorySystem interface
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        prompt = PROMPT_TEMPLATE.format(
            lexicon_section=self._lexicon_section(input),
            examples_section=self._examples_section(input),
            input=input,
        )
        response = self.call_llm(prompt)
        answer = self._canonicalize(extract_json_field(response, "final_answer"))
        retried = False
        if not answer:
            retried = True
            retry_response = self.call_llm(RETRY_TEMPLATE.format(input=input))
            answer = self._canonicalize(extract_json_field(retry_response, "final_answer"))
            response = response + "\n" + retry_response
        return answer, {
            "full_response": response,
            "num_lexicon": len(self.lexicon),
            "num_payload_rules": sum(len(v) for v in self.payload_rules.values()),
            "num_part_rules": sum(len(v) for v in self.part_rules.values()),
            "num_examples": len(self.examples),
            "retried": retried,
        }

    def get_context_length(self) -> int:
        return len(self._lexicon_section("")) + MAX_EXAMPLES * (MAX_EXAMPLE_CHARS + 40)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> str:
        return json.dumps(
            {
                "version": _STATE_VERSION,
                "lexicon": self.lexicon,
                "payload_rules": {k: dict(v) for k, v in self.payload_rules.items()},
                "part_rules": {k: dict(v) for k, v in self.part_rules.items()},
                "examples": self.examples,
                "learned_items": self._learned_items,
            },
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.lexicon = {k: int(v) for k, v in data.get("lexicon", {}).items()}
        self.payload_rules = defaultdict(lambda: defaultdict(int))
        for k, row in data.get("payload_rules", {}).items():
            for target, cnt in row.items():
                self.payload_rules[k][target] = int(cnt)
        self.part_rules = defaultdict(lambda: defaultdict(int))
        for k, row in data.get("part_rules", {}).items():
            for target, cnt in row.items():
                self.part_rules[k][target] = int(cnt)
        self.examples = list(data.get("examples", []))
        self._learned_items = int(data.get("learned_items", len(self.examples)))
