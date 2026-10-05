# Prototype: contrastive confusion memory.
# Self-contained: fake LLM + inline literals copied from the permitted diagnostics views.
# No file/network/process I/O; imports limited to collections.

from collections import defaultdict

# --- literals copied from the permitted diagnostics views (input -> stored label) ---
TRAIN = [
    ("事实:被告人武某甲伙同申某、石某、穆某违反土地管理法规，未经审批擅自在山上雇佣他人挖山采石，致使大面积林地被毁坏。经鉴定，被毁坏的林地面积为12.1亩。",
     "非法占用农用地",
     "[罪名]破坏森林资源<eoa>"),
    ("事实:2014年2月13日至17日凌晨，被告人范某组织周某某等人多次用麻将牌进行二八杠赌博，每场输赢数万元，被告人范某不仅参与赌博，还抽头渔利一万余元。",
     "赌博",
     "[罪名]组织、领导赌博活动<eoa>"),
    ("事实:被告人王某在卓资县卓资山镇和平村委会非法占用农用地采金，三块共计0.9336公顷（折合14.004亩），全部为基本农田。造成被占用的基本农田毁坏。",
     "非法占用农用地",
     "[罪名]非法占用农用地<eoa>"),
    ("事实:被告人张某为向某公司收取工程款项，先后从代开发票小广告处购买了2张税务机关代开统一发票（发票金额合计708433元），并将发票提供给该公司入账。经鉴定，以上发票系假发票。",
     "虚开发票",
     "[罪名]伪造发票罪<eoa>"),
]

# --- queries drawn from the same visible distribution (not from TRAIN) ---
QUERIES = [
    "事实:被告人某某违反土地管理法规，未经审批擅自开垦林地用于种植农作物，造成林地被大量毁坏。经鉴定，被毁坏林地面积为15亩。",
    "事实:被告人某某在其经营的饭馆二楼设置多台麻将桌供他人赌博，从中抽水牟利，赌资累计数额较大。",
    "事实:被告人某某在明知他人贩卖的是伪造的发票的情况下，仍从对方处购买伪造的普通发票若干份。",
]


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def tokenize(text):
    """CJK bigrams + latin/digit words. Hand-rolled to avoid an `re` import."""
    out = set()
    cur = []
    for ch in text:
        if ch.isascii() and ch.isalnum():
            cur.append(ch.lower())
        else:
            if len(cur) >= 2:
                out.add("".join(cur))
            cur = []
    if len(cur) >= 2:
        out.add("".join(cur))
    cjk = [c for c in text if "一" <= c <= "鿿"]
    for i in range(len(cjk) - 1):
        out.add(cjk[i] + cjk[i + 1])
    return frozenset(out)


class FakeLLM:
    """Returns a different JSON payload per call so grouping is observable."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        n = len(self.calls)
        if "reject_label" in prompt:
            return (
                '{"shared_scenario": "unauthorized land use", '
                '"reject_label": "%s", "reject_reason": "r%d", '
                '"prefer_label": "prefer", "prefer_reason": "p%d", '
                '"discriminator": "if the conduct is land occupation, use the land charge; '
                'if it is gambling organisation, use the gambling charge"}' % (n, n, n)
            )
        return '{"reasoning": "r", "final_answer": "[罪名]非法占用农用地<eoa>"}'


class ContrastiveMemory:
    def __init__(self, llm):
        self._llm = llm
        self.episodes = []          # {"input","target","error_lab","tokens"}
        self.confusion = defaultdict(lambda: defaultdict(int))
        self.by_target = defaultdict(list)
        self._lesson_cache = {}     # (bucket, n) -> lesson text

    def _bucket(self, query):
        return frozenset(list(tokenize(query))[:24])

    def _ranked(self, query):
        qt = tokenize(query)
        scored = [(jaccard(qt, ep["tokens"]), i, ep) for i, ep in enumerate(self.episodes)]
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return scored

    def _group_key(self, query):
        """Coarse token bucket; two errors land in the same group only if they are
        near-duplicates, so one lesson call amortises over a genuine cluster."""
        return frozenset(sorted(tokenize(query))[:12])

    def _lesson_for_group(self, members):
        """One LLM call per (group, size). Lazily built, cached."""
        key = (members[0]["bucket"], len(members))
        if key in self._lesson_cache:
            return self._lesson_cache[key]
        lines = []
        for n, ep in enumerate(members):
            lines.append(
                "case%d: %s\nmis-selected: %s\ncorrect: %s"
                % (n + 1, ep["input"][:220], ep["error_lab"] or "(empty)", ep["target"])
            )
        prompt = (
            "You are auditing charge-selection errors made on Chinese criminal cases.\n"
            "Below are cases where a specific charge was mis-selected instead of the "
            "correct one.\n\n"
            + "\n\n".join(lines)
            + "\n\nThese cases were answered incorrectly in the same way. Name the "
            "single conflict this cluster illustrates, then state a decision rule that "
            "tells the two charges apart, phrased so it applies to any future case of "
            "this kind, not just these.\n\n"
            'Respond in JSON: {"shared_scenario": "...", "reject_label": "...", '
            '"reject_reason": "...", "prefer_label": "...", "prefer_reason": "...", '
            '"discriminator": "..."}'
        )
        response = self._llm(prompt)
        text = response.strip()
        self._lesson_cache[key] = text
        return text

    def learn(self, batch):
        for r in batch:
            q = r.get("raw_question", r["input"])
            ep = {
                "input": q,
                "target": r["ground_truth"],
                "error_lab": "" if r.get("was_correct", True) else r.get("prediction", ""),
                "tokens": tokenize(q),
                "bucket": self._group_key(q),
            }
            self.episodes.append(ep)
            if not r.get("was_correct", True):
                self.confusion[r.get("prediction", "")][r["ground_truth"]] += 1

    def build_context(self, query):
        if not self.episodes:
            return "", 0
        ranked = self._ranked(query)
        parts = []

        # Phase 1: contrastive clusters near this query (mechanism under test)
        seen_buckets = []
        members = []
        for score, idx, ep in ranked:
            if not ep["error_lab"]:
                continue
            if ep["bucket"] in seen_buckets:
                continue
            seen_buckets.append(ep["bucket"])
            members.append(ep)
            if len(members) >= 3:
                break
        if members:
            parts.append("[ERROR CONTRASTS]\n" + self._lesson_for_group(members))

        # Phase 2: similarity fill
        for score, idx, ep in ranked[:4]:
            parts.append("Q: %s\nA: %s" % (ep["input"][:200], ep["target"]))
        return "\n\n".join(parts), len(members)


def main():
    llm = FakeLLM()
    mem = ContrastiveMemory(llm)

    print("=== cold start ===")
    ctx, n = mem.build_context(QUERIES[0])
    print("context empty:", ctx == "", "| calls:", len(llm.calls))

    mem.learn([
        {"input": t, "raw_question": t, "ground_truth": g, "prediction": p,
         "was_correct": (p == g)}
        for t, g, p in TRAIN
    ])
    print("\n=== after learning", len(TRAIN), "examples ===")
    print("confusion:", {k: dict(v) for k, v in mem.confusion.items()})

    for i, q in enumerate(QUERIES):
        llm.calls.clear()
        ctx, n = mem.build_context(q)
        head = ctx.split("\n")[0] if ctx else ""
        print("\n--- query %d ---" % i)
        print("contrastive clusters:", n, "| llm calls:", len(llm.calls))
        print("head:", head)
        print("context chars:", len(ctx))

    # Re-query: cache must prevent repeat lesson calls
    llm.calls.clear()
    mem.build_context(QUERIES[0])
    print("\nsecond call on query 0 -> llm calls:", len(llm.calls), "(expect 0)")

    # Distinct groups must produce distinct lesson calls
    llm.calls.clear()
    mem.build_context("事实:被告人某某走私废物入境，逃避海关监管。")
    print("unrelated query -> llm calls:", len(llm.calls))


main()
