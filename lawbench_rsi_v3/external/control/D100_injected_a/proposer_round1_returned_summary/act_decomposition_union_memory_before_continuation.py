"""Act Decomposition Union Memory — charge each act separately, then union system-side.

Diagnosis this addresses (two buckets the base system loses, distinct from the
naming bucket that Canonical Charge Lexicon Memory attacks):

  * Multi-charge undercount. When a target holds 2-3 charges the system answers
    with 1, because a single call over a long narrative collapses every act into
    the one most salient label:
        盗窃              vs 破坏交通设施;盗窃
        盗窃              vs 抢劫;盗窃;抢夺
        伪造货币          vs 伪造货币;故意伤害
  * Over-prediction. Charging the whole narrative at once also invites extras, and
    the system never learns how large the answer is supposed to be:
        诈骗;敲诈勒索;非法拘禁 vs 非法拘禁

Mechanism (an added stage, not a parameter change): the fact is *segmented* into
acts on structural markers, each act is charged in its own short call, and the
system assembles the union itself in first-appearance order. Two pieces of learned
*structural* memory govern the assembly:

  1. Spurious charges — names the model emitted that the truth did not contain —
     strip inventions that are repeatedly produced and never confirmed.
  2. Answer-cardinality prior — the observed distribution of |truth| conditioned on
     how many acts were detected — trims a union that is longer than this kind of
     fact ever warrants, dropping the weakest-evidence excess first.

Note what this system does *not* do: it never rewrites a charge string into another
string, and it keeps no inventory of statutory names. Its memory is about how many
charges an answer should contain and which ones are inventions, not about how a
charge is spelled. Memory of set *structure*, rather than memory of *strings*, is
the axis.

Segmenting also shortens the prompt: a long fact that pushes the answer out of the
response channel is charged act-by-act instead, and an empty act gets one terse
retry.
"""

import re
from collections import defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# One call per act; beyond this the tail is joined into the last call so the call
# budget stays bounded.
MAX_ACTS = 3

ACT_PROMPT = """下面是一起案件事实的其中一部分。请只针对这部分列出它构成的罪名。

**这部分事实:**
{act}

**要求:**
- 用法定罪名，多个罪名用 ';' 连接。
- 只写这部分事实直接支持的罪名；不要补充其他部分可能涉及的罪名。
- 罪名只写在 [罪名] 和 <eoa> 之间。
{examples_section}
{{"reasoning": "简要分析", "final_answer": "[罪名]<罪名><eoa>"}}"""

# Terse retry, used only when an act's first call yields nothing.
RETRY_PROMPT = """这部分事实构成什么罪名？一行作答。

{act}

格式: {{"final_answer": "[罪名]<罪名><eoa>"}}"""

MIN_FACT_CHARS = 240   # below this, segmentation is pointless
MIN_ACT_CHARS = 50     # drop fragments too short to carry a charge
SPURIOUS_MIN = 2       # unconfirmed inventions must recur before being stripped
MIN_GLOBAL_OBS = 10    # observations before the global cardinality prior is trusted
MAX_EXAMPLES = 2
MAX_EXAMPLE_CHARS = 180
_STATE_VERSION = 1

# Structural markers that introduce a distinct act: 一、/（一）/(一)/1、/1./1．
_MARKER = re.compile(
    r"(?m)^[ \t　]*(?:[一二三四五六七八九十]{1,2}[ \t]*、"
    r"|（[一二三四五六七八九十]{1,2}）"
    r"|\([一二三四五六七八九十]{1,2}\)"
    r"|\d{1,2}[ \t]*[、.．])"
)
_PARAGRAPH = re.compile(r"\n{2,}")


def _clean(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _split_charges(s: str) -> list[str]:
    """Split on ';' only — '、' is internal to statutory names and must not split."""
    return [p.strip() for p in re.split(r"[;；]", s or "") if p.strip()]


def _payload(s: str) -> str:
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


def _segment_fact(
    text: str, max_acts: int = MAX_ACTS, min_fact: int = MIN_FACT_CHARS, min_act: int = MIN_ACT_CHARS
) -> list[str]:
    """Split a fact into acts on structural markers, falling back to paragraphs.

    Returns [whole_text] when nothing splits cleanly, so single-act facts are
    charged exactly as before.
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return []
    if len(t) < min_fact:
        return [t]
    for pattern in (_MARKER, _PARAGRAPH):
        starts = [0] + [m.start() for m in pattern.finditer(t) if m.start() > 0]
        segs = [t[starts[i] : starts[i + 1]].strip() for i in range(len(starts) - 1)]
        segs = [s for s in segs if len(s) >= min_act]
        if len(segs) >= 2:
            if len(segs) > max_acts:
                segs = segs[: max_acts - 1] + ["\n".join(segs[max_acts - 1 :])]
            return segs
    return [t]


class ActDecompositionUnionMemory(MemorySystem):
    """Per-act charging with system-side union and learned charge-count control."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        # charge predicted but absent from the truth -> times observed
        self.spurious: dict[str, int] = defaultdict(int)
        # charge that appeared in some truth -> times observed
        self.confirmed: dict[str, int] = defaultdict(int)
        # number of detected acts -> {number of true charges: times observed}
        self.size_by_acts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # every observed |truth|, for facts whose act count was never seen
        self.global_sizes: dict[int, int] = defaultdict(int)
        self._size_obs = 0
        # (act text, charge) style hints, most recent last
        self.examples: list[dict[str, str]] = []
        self._learned_items = 0

    # ------------------------------------------------------------------
    # Learning — structural memory only; no string rewriting
    # ------------------------------------------------------------------

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            truth = _clean(_payload(r.get("ground_truth", "")))
            if not truth:
                continue
            truth_parts = [_clean(t) for t in _split_charges(truth)]
            truth_parts = [t for t in truth_parts if t]
            if not truth_parts:
                continue
            for t in truth_parts:
                self.confirmed[t] += 1

            truth_set = set(truth_parts)
            pred_parts = [_clean(p) for p in _split_charges(_payload(r.get("prediction", "")))]
            for p in pred_parts:
                if p and p not in truth_set:
                    self.spurious[p] += 1

            fact = r.get("raw_question", r.get("input", "")) or ""
            acts = _segment_fact(fact)
            self.size_by_acts[len(acts)][len(truth_parts)] += 1
            self.global_sizes[len(truth_parts)] += 1
            self._size_obs += 1

            self.examples.append({"fact": _clean(fact)[:MAX_EXAMPLE_CHARS], "target": truth})
            self._learned_items += 1

        if len(self.examples) > MAX_EXAMPLES * 4:
            self.examples = self.examples[-MAX_EXAMPLES * 4 :]

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _expected_size(self, n_acts: int) -> int | None:
        """Most common true charge count for a fact with this many acts."""
        row = self.size_by_acts.get(n_acts)
        if row:
            return max(row.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if self._size_obs >= MIN_GLOBAL_OBS and self.global_sizes:
            return max(self.global_sizes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        return None

    def _calibrate(self, charges: list[str], n_acts: int) -> list[str]:
        if not charges:
            return []
        # Layer 1: strip inventions that recur without ever being confirmed.
        kept = [
            c
            for c in charges
            if not (self.spurious.get(c, 0) >= SPURIOUS_MIN and c not in self.confirmed)
        ]
        if not kept:
            kept = list(charges)
        # Layer 2: an answer longer than this kind of fact ever warrants loses its
        # weakest-evidence parts first (unconfirmed before confirmed).
        expected = self._expected_size(n_acts)
        if expected is not None and expected >= 1:
            while len(kept) > expected and len(kept) > 1:
                idx = max(
                    range(len(kept)),
                    key=lambda i: (
                        self.spurious.get(kept[i], 0),
                        1 if kept[i] not in self.confirmed else 0,
                    ),
                )
                kept.pop(idx)
        return kept

    def _parse(self, response: str) -> list[str]:
        out, seen = [], set()
        for p in _split_charges(_payload(extract_json_field(response, "final_answer"))):
            c = _clean(p)
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _examples_section(self, act: str) -> str:
        if not self.examples:
            return ""
        ranked = sorted(self.examples, key=lambda ex: -_sim(act, ex["fact"]))[:MAX_EXAMPLES]
        blocks = [f"例：{ex['fact']}\n罪名：{ex['target']}" for ex in ranked]
        return "\n" + "\n".join(blocks) + "\n"

    # ------------------------------------------------------------------
    # MemorySystem interface
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        acts = _segment_fact(input)
        if not acts:
            return "", {"full_response": "", "num_acts": 0, "retried": False}

        responses: list[str] = []
        per_act: list[list[str]] = []
        retries = 0
        for act in acts:
            response = self.call_llm(ACT_PROMPT.format(act=act, examples_section=self._examples_section(act)))
            parts = self._parse(response)
            if not parts:
                retries += 1
                retry_response = self.call_llm(RETRY_PROMPT.format(act=act))
                response = response + "\n" + retry_response
                parts = self._parse(retry_response)
            responses.append(response)
            per_act.append(parts)

        union: list[str] = []
        for parts in per_act:
            for c in parts:
                if c not in union:
                    union.append(c)
        kept = self._calibrate(union, len(acts))
        answer = _wrap(";".join(kept)) if kept else ""

        return answer, {
            "full_response": "\n".join(responses),
            "num_acts": len(acts),
            "per_act": per_act,
            "union": union,
            "kept": kept,
            "expected_size": self._expected_size(len(acts)),
            "retries": retries,
            "num_spurious": len(self.spurious),
            "size_observations": self._size_obs,
        }

    def get_context_length(self) -> int:
        return MAX_ACTS * 900

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> str:
        import json

        return json.dumps(
            {
                "version": _STATE_VERSION,
                "spurious": dict(self.spurious),
                "confirmed": dict(self.confirmed),
                "size_by_acts": {str(k): dict(v) for k, v in self.size_by_acts.items()},
                "global_sizes": dict(self.global_sizes),
                "size_obs": self._size_obs,
                "examples": self.examples,
                "learned_items": self._learned_items,
            },
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        import json

        data = json.loads(state)
        self.spurious = defaultdict(int, {k: int(v) for k, v in data.get("spurious", {}).items()})
        self.confirmed = defaultdict(int, {k: int(v) for k, v in data.get("confirmed", {}).items()})
        self.size_by_acts = defaultdict(lambda: defaultdict(int))
        for k, row in data.get("size_by_acts", {}).items():
            for size, cnt in row.items():
                self.size_by_acts[int(k)][int(size)] = int(cnt)
        self.global_sizes = defaultdict(int, {int(k): int(v) for k, v in data.get("global_sizes", {}).items()})
        self._size_obs = int(data.get("size_obs", 0))
        self.examples = list(data.get("examples", []))
        self._learned_items = int(data.get("learned_items", len(self.examples)))
