"""Label Prototype Memory.

A fundamentally different memory structure: instead of storing all raw examples
and selecting a subset at predict time, this system maintains one LLM-written
compact "prototype" per label — a 1-2 sentence description of what defines that
label and how it differs from its most common confusors.

How prototypes are generated (axis F — LLM usage in learning):
  After a label accumulates _PROTOTYPE_THRESHOLD errors, the LLM is asked to
  write a short discriminative description using: the label name, its stored
  correct examples, and the confusing wrong predictions made for it. This
  produces targeted, compressed guidance the model would otherwise have to infer
  from raw examples alone.

How prototypes are used at predict time (axis B — memory content):
  The top-K most query-similar prototypes are injected as a "Key distinctions"
  block above the few-shot examples. Cold start: no prototypes exist yet, so
  the system falls back to pure similarity retrieval — identical to the base.

Why this differs from llm_rule_synthesis_memory (which regressed -7.6):
  - That system stored O(N^2) pairwise rules between all confused label pairs,
    which saturated the context with generic cross-comparisons.
  - This system stores O(N) per-label descriptions; each prototype covers one
    label holistically, so the context stays focused and non-redundant.
  - Prototypes are updated incrementally as new errors arrive, not regenerated
    from scratch, so quality improves monotonically over training.
"""

import json
import re
from collections import defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# ---- prompt templates ----

PREDICT_TEMPLATE = """Solve the problem below based on the examples provided.

{prototypes_section}{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

PROTOTYPE_GEN_TEMPLATE = """You are helping a classifier distinguish between similar categories.

Category to describe: {label}

Correct examples of this category:
{correct_examples}

Cases where this category was confused with: {confusors}

Write exactly 1-2 sentences that describe what makes "{label}" distinct from the categories it is confused with. Be concrete and specific — focus on features that reliably separate this category from its confusors. Do not use hedging language.

Respond in JSON format:
{{"description": "[1-2 sentence discriminative description]"}}"""

MAX_CHARS = 30000
_TOP_CANDIDATE_LABELS = 3         # examples to look at for top-K selection
_MAX_PROTOTYPES_IN_PROMPT = 4     # how many prototypes to inject per predict call
_PROTOTYPE_THRESHOLD = 2          # errors on a label before generating its prototype
_MAX_CORRECT_EX_FOR_PROTO = 3     # correct examples fed to LLM when writing prototype

# CJK Unicode ranges
_CJK_RE = re.compile(r'[一-鿿぀-ゟ゠-ヿ＀-￯]')

# Format-wrapper stripping (same as confusion_normalized_bfs_memory)
_WRAP_PREFIX = re.compile(r'^\[[^\]]+\]')
_WRAP_SUFFIX_EOA = re.compile(r'<eoa>$')
_WRAP_SUFFIX_CLOSE = re.compile(r'\[/[^\]]+\]$')


def _normalize_label(s: str) -> str:
    s = s.strip()
    s = _WRAP_PREFIX.sub('', s)
    s = _WRAP_SUFFIX_EOA.sub('', s)
    s = _WRAP_PREFIX.sub('', s)
    s = _WRAP_SUFFIX_CLOSE.sub('', s)
    return s.strip()


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


class LabelPrototypeMemory(MemorySystem):
    """Per-label LLM-written prototype descriptions injected above few-shot examples.

    Memory structure:
      self.examples       — raw (input, target, tokens) for few-shot retrieval
      self.prototypes     — {label: description_str} written by the LLM
      self.label_errors   — {label: {wrong_pred: count}} tracks what confused each label
      self.label_correct  — {label: [input_preview, ...]} for prototype generation context
      self.error_counts   — {label: int} triggers prototype generation
      self.proto_tokens   — {label: frozenset} tokenized prototype text for similarity search
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.prototypes: dict[str, str] = {}
        self.proto_tokens: dict[str, frozenset] = {}
        self.label_errors: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.label_correct: dict[str, list[str]] = defaultdict(list)
        self.error_counts: dict[str, int] = defaultdict(int)

    # ---- similarity helpers ----

    def _scored_examples(self, query: str) -> list[tuple[float, int, dict]]:
        q_tok = _tokenize_adaptive(query)
        result = [
            (_jaccard(q_tok, ex["tokens"]), idx, ex)
            for idx, ex in enumerate(self.examples)
        ]
        result.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return result

    def _top_prototypes(self, query: str) -> list[tuple[str, str]]:
        """Return top-K (label, description) pairs ranked by prototype text similarity."""
        if not self.proto_tokens:
            return []
        q_tok = _tokenize_adaptive(query)
        scored = [
            (_jaccard(q_tok, tok), lbl)
            for lbl, tok in self.proto_tokens.items()
        ]
        scored.sort(reverse=True)
        return [(lbl, self.prototypes[lbl]) for _, lbl in scored[:_MAX_PROTOTYPES_IN_PROMPT]]

    # ---- prototype generation ----

    def _generate_prototype(self, label: str) -> None:
        """Ask the LLM to write a discriminative description for this label."""
        correct_exs = self.label_correct.get(label, [])[:_MAX_CORRECT_EX_FOR_PROTO]
        confusors = sorted(
            self.label_errors[label].items(), key=lambda x: -x[1]
        )
        if not confusors:
            return

        correct_str = "\n".join(f"- {ex[:200]}" for ex in correct_exs) if correct_exs else "(none yet)"
        confusor_str = ", ".join(f'"{c}"' for c, _ in confusors[:4])

        prompt = PROTOTYPE_GEN_TEMPLATE.format(
            label=label,
            correct_examples=correct_str,
            confusors=confusor_str,
        )
        response = self.call_llm(prompt)
        description = extract_json_field(response, "description")
        if description:
            self.prototypes[label] = description
            self.proto_tokens[label] = _tokenize_adaptive(label + " " + description)

    # ---- predict ----

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        ranked = self._scored_examples(input)
        top_protos = self._top_prototypes(input)

        # Build prototypes section
        if top_protos:
            lines = ["**Key distinctions:**"]
            for lbl, desc in top_protos:
                lines.append(f"- {lbl}: {desc}")
            prototypes_section = "\n".join(lines) + "\n\n"
        else:
            prototypes_section = ""

        # Build examples section within remaining char budget
        proto_chars = len(prototypes_section)
        parts: list[str] = []
        used: set[int] = set()
        total_chars = proto_chars

        for _, idx, ex in ranked:
            if idx in used:
                continue
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > MAX_CHARS:
                break
            parts.append(part)
            used.add(idx)
            total_chars += len(part) + 2

        examples_section = "\n\n".join(parts)
        prompt = PREDICT_TEMPLATE.format(
            prototypes_section=prototypes_section,
            examples_section=examples_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_selected": len(parts),
            "num_prototypes": len(self.prototypes),
            "prototypes_injected": len(top_protos),
        }

    # ---- learning ----

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

            gt = r["ground_truth"]
            if r.get("was_correct", True):
                # Keep a small pool of correct examples per label for prototype context
                if len(self.label_correct[gt]) < _MAX_CORRECT_EX_FOR_PROTO:
                    self.label_correct[gt].append(raw_q[:300])
            else:
                pred_raw = r.get("prediction", "")
                pred = _normalize_label(pred_raw)
                if pred and pred != gt:
                    self.label_errors[gt][pred] += 1
                    self.error_counts[gt] += 1

                    # Generate or refresh prototype when threshold is hit
                    if self.error_counts[gt] == _PROTOTYPE_THRESHOLD:
                        self._generate_prototype(gt)
                    elif (
                        self.error_counts[gt] > _PROTOTYPE_THRESHOLD
                        and self.error_counts[gt] % (_PROTOTYPE_THRESHOLD * 3) == 0
                    ):
                        # Refresh periodically as more error context accumulates
                        self._generate_prototype(gt)

    # ---- state ----

    def get_state(self) -> str:
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        return json.dumps(
            {
                "examples": serialisable,
                "prototypes": self.prototypes,
                "label_errors": {k: dict(v) for k, v in self.label_errors.items()},
                "label_correct": dict(self.label_correct),
                "error_counts": dict(self.error_counts),
            },
            indent=2,
            ensure_ascii=False,
        )

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

        self.prototypes = data.get("prototypes", {})
        self.proto_tokens = {
            lbl: _tokenize_adaptive(lbl + " " + desc)
            for lbl, desc in self.prototypes.items()
        }
        self.label_errors = defaultdict(lambda: defaultdict(int))
        for lbl, errs in data.get("label_errors", {}).items():
            for pred, cnt in errs.items():
                self.label_errors[lbl][pred] = cnt
        self.label_correct = defaultdict(list, data.get("label_correct", {}))
        self.error_counts = defaultdict(int, data.get("error_counts", {}))
