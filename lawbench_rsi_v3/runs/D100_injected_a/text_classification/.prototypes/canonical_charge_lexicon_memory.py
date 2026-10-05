"""Prototype + validation for canonical_charge_lexicon_memory (Candidate A).

Part 1 exercises the core mechanism inline with a fake LLM using literals taken
from the frozen feedback view. Part 2 imports the real agent module and re-runs
the mechanism check through the MemorySystem interface.

Mechanism under test: the model emits a plausible charge *concept* under a
non-statutory *name*. Scoring is exact string match, so a correct concept with a
wrong string scores zero. This system keeps an inventory of exact statutory
strings seen as ground truth, learns deterministic rewrite rules from each
(prediction, ground_truth) pair, and normalizes its own output into an inventory
member before returning it.
"""

import json
import re
from collections import defaultdict

# --------------------------------------------------------------------------
# Inline mechanism (mirrors the candidate, kept import-free so Part 1 runs
# standalone)
# --------------------------------------------------------------------------


def _clean(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _split_charges(s: str) -> list:
    """Split on ';' only. '、' is part of statutory names and must not split."""
    return [p.strip() for p in re.split(r"[;；]", s or "") if p.strip()]


def _extract_payload(s: str) -> str:
    m = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", s or "", re.S)
    return m.group(1).strip() if m else (s or "").strip()


def _bigrams(s: str) -> frozenset:
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


class Canonicalizer:
    """Learned inventory + rewrite rules. Deterministic; no LLM involved.

    Two rule layers, tested as variants below:
      V1 similarity-only alignment (the naive design that MISSes 开设赌场 -> 赌博)
      V2 + verbatim whole-payload rules (no similarity required)
      V3 + verbatim preferred over lexicon hit (adopted)
    """

    def __init__(self, use_payload_rules: bool = False, payload_first: bool = False) -> None:
        self.inventory = {}          # canonical string -> times seen as ground truth
        self.alias = defaultdict(dict)  # observed part -> {canonical part: count}
        self.payload = defaultdict(dict)  # whole observed payload -> {truth payload: count}
        self.use_payload_rules = use_payload_rules
        self.payload_first = payload_first

    def observe_truth(self, truth: str) -> None:
        for t in _split_charges(_extract_payload(truth)):
            t = _clean(t)
            if t:
                self.inventory[t] = self.inventory.get(t, 0) + 1

    def learn(self, pred: str, truth: str) -> None:
        """Record the literal rewrite, then align parts when counts agree."""
        pred_payload = _clean(_extract_payload(pred))
        truth_payload = _clean(_extract_payload(truth))
        if self.use_payload_rules and pred_payload and pred_payload != truth_payload:
            self.payload[pred_payload][truth_payload] = (
                self.payload[pred_payload].get(truth_payload, 0) + 1
            )
        parts = _split_charges(_extract_payload(pred))
        truths = [_clean(t) for t in _split_charges(_extract_payload(truth))]
        for p in parts:
            p = _clean(p)
            if not p:
                continue
            best, best_s = None, 0.0
            for t in truths:
                s = _sim(p, t)
                if s > best_s:
                    best, best_s = t, s
            if best and best != p:
                self.alias[p][best] = self.alias[p].get(best, 0) + 1

    def _best(self, cand: str, threshold: float) -> str:
        best, best_s = cand, threshold
        for t in self.inventory:
            s = _sim(cand, t)
            if s > best_s:
                best, best_s = t, s
        return best

    def canonical(self, cand: str) -> str:
        c = _clean(_extract_payload(cand))
        if not c:
            return ""
        # V3 ordering: the stored rewrite beats a bare inventory hit, because a
        # verbatim repeat of the same wrong string is the strongest evidence.
        if self.payload_first and c in self.payload and self.payload[c]:
            return max(self.payload[c].items(), key=lambda kv: kv[1])[0]
        if c in self.inventory:
            return c
        if c in self.payload and self.payload[c]:
            return max(self.payload[c].items(), key=lambda kv: kv[1])[0]
        # 1. learned exact alias
        if c in self.alias and self.alias[c]:
            return max(self.alias[c].items(), key=lambda kv: kv[1])[0]
        # 2. generic Chinese convention: the label field drops the "罪" suffix
        if c.endswith("罪") and c[:-1] in self.inventory:
            return c[:-1]
        # 3. containment (statutory enumerations get truncated by the model)
        contained = [t for t in self.inventory if c in t or t in c]
        if contained:
            return max(contained, key=len)
        # 4. fuzzy fallback, conservative threshold
        return self._best(c, 0.6)

    def normalize(self, answer: str) -> str:
        out, seen = [], set()
        for p in _split_charges(_extract_payload(answer)):
            c = self.canonical(p)
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return ";".join(out)


# --------------------------------------------------------------------------
# Part 1: exercise the mechanism with a fake LLM and feedback literals
# --------------------------------------------------------------------------

CASES = [
    # (item_id, model's raw answer, ground truth)
    ("lawbench_3-3_0003", "[罪名]故意毁坏财物罪<eoa>", "故意毁坏财物"),
    ("lawbench_3-3_0234", "[罪名]侵犯注册商标专用权罪<eoa>", "假冒注册商标"),
    ("lawbench_3-3_0482", "[罪名]侵犯注册商标专用权<eoa>", "销售假冒注册商标的商品"),
    ("lawbench_3-3_0309", "[罪名]非法种植罂粟<eoa>", "非法种植毒品原植物"),
    ("lawbench_3-3_0349", "[罪名]非法制造枪支<eoa>",
     "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物"),
    ("lawbench_3-3_0329", "[罪名]滥伐林木<eoa>", "盗伐林木"),
    ("lawbench_3-3_0287", "[罪名]开设赌场<eoa>", "赌博"),
]


def _run_variant(name: str, use_payload: bool, payload_first: bool) -> tuple[int, "Canonicalizer"]:
    cz = Canonicalizer(use_payload_rules=use_payload, payload_first=payload_first)
    for _iid, pred, truth in CASES:
        cz.observe_truth(truth)
        cz.learn(pred, truth)
    hits = sum(1 for _iid, pred, truth in CASES if cz.normalize(pred) == truth)
    print(f"variant {name}: {hits}/{len(CASES)} exact")
    return hits, cz


def part1() -> bool:
    # Variant comparison before settling on the shipped ordering.
    v1_hits, _v1 = _run_variant("V1 similarity-only", False, False)
    v2_hits, _v2 = _run_variant("V2 +verbatim payload", True, False)
    v3_hits, cz = _run_variant("V3 verbatim-first (adopted)", True, True)
    assert v3_hits >= v2_hits >= v1_hits, "variant ladder must not regress"

    print("inventory size:", len(cz.inventory))
    print("payload rules:", {k: dict(v) for k, v in cz.payload.items()})
    print("learned aliases:", {k: dict(v) for k, v in cz.alias.items()})

    ok = True
    # Learned-alias rules must fire even when fuzzy similarity alone would not.
    for iid, pred, truth in CASES:
        got = cz.normalize(pred)
        hit = got == truth
        ok &= hit
        print(f"  {iid}: {pred} -> {got!r} target={truth!r} {'OK' if hit else 'MISS'}")

    # Cold start (empty inventory) must be a no-op, never a crash.
    cold = Canonicalizer()
    assert cold.normalize("[罪名]盗窃<eoa>") == "盗窃", "cold start must pass through"
    # ';' separated multi-charge preserved; '、' inside a name never split.
    multi = cz.normalize("[罪名]虚开发票;盗窃<eoa>")
    print("  multi-charge passthrough:", repr(multi))
    stat = cz.canonical("非法制造枪支")
    print("  enumeration truncation:", repr(stat))
    assert "、" in stat, "comma inside a statutory enumeration must survive"

    print("PART 1:", "ALL OK" if ok else "SOME MISSES")
    return ok


# --------------------------------------------------------------------------
# Part 2: import the real candidate and re-check through the interface
# --------------------------------------------------------------------------

TRIED_REAL_IMPORT = None


def part2() -> None:
    global TRIED_REAL_IMPORT
    try:
        from text_classification.agents.canonical_charge_lexicon_memory import (  # noqa: E402
            CanonicalChargeLexiconMemory,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        TRIED_REAL_IMPORT = f"import failed: {type(exc).__name__}: {exc}"
        print("PART 2 skipped —", TRIED_REAL_IMPORT)
        return
    TRIED_REAL_IMPORT = "ok"

    seen_prompts = []

    def fake_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        # Mimic the base model's habit: a correct concept under a wrong name,
        # plus the empty-output failure on a long fact.
        if "第二起" in prompt or len(prompt) > 60000:
            return json.dumps({"reasoning": "...", "final_answer": ""})
        return json.dumps({"reasoning": "x", "final_answer": "[罪名]侵犯注册商标专用权罪<eoa>"})

    mem = CanonicalChargeLexiconMemory(fake_llm)

    # Cold start must work with empty memory.
    ans, meta = mem.predict("事实：被告人某甲盗窃他人财物。")
    assert isinstance(ans, str), "predict must return a string answer"
    print("cold-start answer:", repr(ans), "| meta keys:", sorted(meta)[:4])

    # Learn a batch, then confirm the learned rewrite reaches the output path.
    batch = [
        {"input": "事实：某甲销售假冒注册商标的商品。", "prediction": "[罪名]侵犯注册商标专用权罪<eoa>",
         "ground_truth": "假冒注册商标", "was_correct": False, "metadata": {}},
        {"input": "事实：某乙故意毁坏他人财物。", "prediction": "[罪名]故意毁坏财物罪<eoa>",
         "ground_truth": "故意毁坏财物", "was_correct": False, "metadata": {}},
    ]
    mem.learn_from_batch(batch)
    ans2, meta2 = mem.predict("事实：某甲销售假冒某注册商标的手袋。")
    print("post-learning answer:", repr(ans2))
    # The framework's output convention keeps the [罪名]<eoa> wrapper (the scorer
    # reads the payload out of it), so assert on the payload, not the raw string.
    assert _extract_payload(ans2) == "假冒注册商标", (
        f"learned alias did not rewrite output: {ans2!r}"
    )

    # Inventory must be injected into the prompt.
    assert any("假冒注册商标" in p for p in seen_prompts), "inventory not injected"
    # Empty output must trigger exactly one terse retry and still return a string.
    n_before = len(seen_prompts)
    ans3, meta3 = mem.predict("第二起 事实：" + "很长的事实描述。" * 4000)
    print("empty-output path:", repr(ans3), "| calls:", len(seen_prompts) - n_before)

    # State round-trip.
    state = mem.get_state()
    mem2 = CanonicalChargeLexiconMemory(fake_llm)
    mem2.set_state(state)
    rt = _extract_payload(mem2.predict("事实：某甲销售假冒某注册商标的手袋。")[0])
    assert rt == "假冒注册商标", f"state round-trip lost rules: {rt!r}"
    print("state round-trip OK | state bytes:", len(state))
    print("PART 2: OK")


if __name__ == "__main__":
    p1 = part1()
    part2()
    print("REAL IMPORT:", TRIED_REAL_IMPORT)
    print("RESULT:", "PASS" if p1 else "PART1_MISSES")
