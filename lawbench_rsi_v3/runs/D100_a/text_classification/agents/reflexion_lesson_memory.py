"""Reflexion Lesson Memory — LLM-distilled transfer rules from errors.

All prior systems store raw examples and retrieve by similarity. This system
takes a fundamentally different approach inspired by Reflexion (Shinn et al.):

1. During learn_from_batch, for each wrong prediction, call the LLM to
   synthesise a concise, generalisable correction rule from the triple
   (input_excerpt, wrong_prediction, ground_truth).

2. Rules are stored with keyword fingerprints extracted by the LLM.

3. At predict time, retrieve the top-k most relevant rules by bigram
   overlap with the query, then inject them as explicit guidance BEFORE
   the few-shot examples (which are omitted — memory IS the rule set).

Key differences from all prior systems:
- Memory content: distilled rules, not raw examples (axis B)
- Learning trigger: LLM call on every error (axis F)
- Prompt architecture: rules-first guidance section (axis A)
- No similarity-ranked example bank at all

Hypothesis (falsifiable): LLM-synthesised correction rules will achieve
higher val accuracy than 26% by directly addressing the confusion patterns
the model repeatedly makes, without the noise of irrelevant raw examples.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# ── Prompt templates ──────────────────────────────────────────────────────────

_LESSON_PROMPT = """An AI made a classification error. Write a concise correction rule.

Case facts (excerpt, first 600 chars):
{facts}

Wrong prediction: {wrong}
Correct answer: {correct}

Write one general rule (1-2 sentences) that would prevent this class of error.
Also list 3-6 keyword phrases that identify when the rule applies.
Respond only in JSON:

{{"rule": "...", "keywords": ["kw1", "kw2", ...]}}"""

_PREDICT_PROMPT = """Solve the classification problem below.
{rules_section}
**Problem:**
{input}

**Instructions:**
- Apply any relevant rules listed above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_RULES = 60          # max stored rules (oldest dropped when exceeded)
MAX_RULE_CHARS = 200    # truncate very long generated rules
MAX_RULES_IN_PROMPT = 5  # inject at most this many rules per prediction
MAX_CHARS_RULES = 4000  # char budget for rules section


def _tokenize(text: str) -> frozenset:
    """Chinese bigrams + ASCII tokens for keyword matching."""
    ascii_tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    chars = re.findall(r"[一-鿿]", text)
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return frozenset(ascii_tokens + bigrams)


def _rule_tokens(rule: dict) -> frozenset:
    """Combined token set for a rule: its text + all keyword phrases."""
    combined = rule.get("rule", "") + " ".join(rule.get("keywords", []))
    return _tokenize(combined)


def _relevance(query_toks: frozenset, rule_toks: frozenset) -> float:
    if not rule_toks:
        return 0.0
    return len(query_toks & rule_toks) / len(rule_toks)


class ReflexionLessonMemory(MemorySystem):
    """Error-driven lesson memory: distilled rules replace raw examples."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        # Each entry: {"rule": str, "keywords": list[str], "tokens": frozenset}
        self.rules: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str) -> list[dict]:
        """Return top rules by relevance to query, within char budget."""
        if not self.rules:
            return []
        q_toks = _tokenize(query)
        scored = sorted(
            self.rules,
            key=lambda r: _relevance(q_toks, r["tokens"]),
            reverse=True,
        )
        selected: list[dict] = []
        total = 0
        for r in scored:
            if len(selected) >= MAX_RULES_IN_PROMPT:
                break
            snippet = r["rule"][:MAX_RULE_CHARS]
            if total + len(snippet) + 4 > MAX_CHARS_RULES:
                break
            selected.append(r)
            total += len(snippet) + 4
        return selected

    def _distil_rule(self, input_text: str, wrong: str, correct: str) -> dict | None:
        """Call LLM to synthesise a correction rule; return parsed dict or None."""
        facts_excerpt = input_text[:600]
        prompt = _LESSON_PROMPT.format(
            facts=facts_excerpt,
            wrong=wrong,
            correct=correct,
        )
        response = self.call_llm(prompt)

        # Parse the JSON blob from the response
        m = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None

        rule_text = str(data.get("rule", "")).strip()
        keywords = [str(k) for k in data.get("keywords", []) if k]
        if not rule_text:
            return None

        return {
            "rule": rule_text[:MAX_RULE_CHARS],
            "keywords": keywords,
            "tokens": _rule_tokens({"rule": rule_text, "keywords": keywords}),
        }

    # ------------------------------------------------------------------
    # MemorySystem interface
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        relevant = self._retrieve(input)

        if relevant:
            rule_lines = "\n".join(
                f"- {r['rule'][:MAX_RULE_CHARS]}" for r in relevant
            )
            rules_section = f"\n**Correction rules (apply where relevant):**\n{rule_lines}\n\n"
        else:
            rules_section = ""

        prompt = _PREDICT_PROMPT.format(
            rules_section=rules_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_rules": len(self.rules),
            "num_matched": len(relevant),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            if r.get("was_correct", True):
                continue
            pred = r.get("prediction", "")
            gt = r["ground_truth"]
            if not pred or pred == gt:
                continue

            rule = self._distil_rule(r["input"], pred, gt)
            if rule is None:
                continue

            # Deduplicate: skip if an identical rule text already exists
            if any(existing["rule"] == rule["rule"] for existing in self.rules):
                continue

            self.rules.append(rule)
            # Evict oldest when over capacity
            if len(self.rules) > MAX_RULES:
                self.rules = self.rules[-MAX_RULES:]

    def get_state(self) -> str:
        serialisable = [
            {"rule": r["rule"], "keywords": r["keywords"]}
            for r in self.rules
        ]
        return json.dumps({"rules": serialisable}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.rules = []
        for entry in data.get("rules", []):
            rule_text = entry.get("rule", "")
            keywords = entry.get("keywords", [])
            self.rules.append({
                "rule": rule_text,
                "keywords": keywords,
                "tokens": _rule_tokens({"rule": rule_text, "keywords": keywords}),
            })
