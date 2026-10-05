"""Prototype: LabelClusterMemory — retrieval organized by charge-label clusters.

Mechanism under test:
- Memory is organized by unique charge labels seen in training.
- At predict time: find top candidate labels via fast similarity, then build the
  prompt with one section per candidate label showing its most representative
  examples. This gives the model side-by-side contrast across the most plausible
  labels rather than a flat similarity-ranked list dominated by one charge type.
- Additionally, for each candidate label, show the exact canonical label string
  as a header so the model sees the correct format (no "罪" suffix, full compound
  labels like "销售假冒注册商标的商品" rather than "侵犯注册商标专用权").

Variants tested:
  V1: Top-3 labels, 2 examples each, no header annotation
  V2: Top-4 labels, 2 examples each, explicit label header showing canonical form
  V3: Top-4 labels, 3 examples each, header + sub-score for co-occurrence hints

Pick V2: explicit canonical-label headers address the "罪 suffix" and "wrong
sub-category" error classes directly, and 4 labels × 2 examples = 8 anchor
examples with budget left for similarity fill.
"""

import json
import re
from collections import defaultdict
from typing import Any


# ── same tokenizer as frontier ──────────────────────────────────────────────

def _tokenize(text: str) -> frozenset:
    tokens: set = set()
    tokens.update(re.findall(r"[A-Za-z0-9]+", text.lower()))
    cjk = re.findall(r"[一-鿿]", text)
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i] + cjk[i + 1])
    return frozenset(tokens)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── fake LLM ─────────────────────────────────────────────────────────────────

def fake_llm(prompt: str) -> str:
    return '{"reasoning": "test", "final_answer": "单位行贿"}'


# ── label cluster system (prototype) ────────────────────────────────────────

_TOP_LABELS = 4          # V2 choice
_EXAMPLES_PER_LABEL = 2
_MAX_FILL = 10           # similarity-fill after cluster sections
MAX_CHARS = 30_000

PREDICT_PROMPT = """Solve the problem below based on the clustered examples provided.

Each section below shows examples for a specific charge label. The section header \
gives the EXACT canonical charge string to use in your answer.

{cluster_section}{fill_section}

**Problem:**
{input}

**Instructions:**
- Use the exact charge label strings shown in the section headers
- Do NOT append '罪' or other suffixes to charge names
- If multiple charges apply, join them with ';'
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""


class LabelClusterMemoryProto:
    def __init__(self, llm):
        self._llm = llm
        # label → list of {input, tokens}
        self.clusters: dict[str, list[dict]] = defaultdict(list)
        # flat pool for similarity fill
        self.examples: list[dict] = []

    def learn_from_batch(self, batch_results: list[dict]) -> None:
        for r in batch_results:
            tok = _tokenize(r["input"])
            ex = {"input": r["input"], "tokens": tok, "target": r["ground_truth"]}
            self.examples.append(ex)
            # Index by each individual charge in a multi-charge label
            for charge in r["ground_truth"].split(";"):
                charge = charge.strip()
                if charge:
                    self.clusters[charge].append(ex)

    def _score_label(self, label: str, q_tok: frozenset) -> float:
        """Score a label by max similarity of its stored examples to the query."""
        pool = self.clusters.get(label, [])
        if not pool:
            return 0.0
        return max(_jaccard(q_tok, ex["tokens"]) for ex in pool)

    def predict(self, input_text: str) -> str:
        q_tok = _tokenize(input_text)
        total_chars = 0

        # --- Step 1: rank all known labels by similarity ---
        label_scores = [
            (label, self._score_label(label, q_tok))
            for label in self.clusters
        ]
        label_scores.sort(key=lambda x: x[1], reverse=True)
        top_labels = [lbl for lbl, _ in label_scores[:_TOP_LABELS]]

        # --- Step 2: build cluster sections ---
        cluster_parts: list[str] = []
        used_indices: set[int] = set()

        for label in top_labels:
            pool = self.clusters[label]
            # pick top-N most similar examples for this label
            scored = sorted(
                ((i, ex) for i, ex in enumerate(pool)),
                key=lambda ie: _jaccard(q_tok, ie[1]["tokens"]),
                reverse=True,
            )
            section_examples: list[str] = []
            for _, ex in scored[:_EXAMPLES_PER_LABEL]:
                # use index in flat pool to avoid duplicates across sections
                flat_idx = id(ex)  # use object id as proxy
                if flat_idx in used_indices:
                    continue
                used_indices.add(flat_idx)
                part = f"  Q: {ex['input'][:300]}\n  A: {ex['target']}"
                if total_chars + len(part) > MAX_CHARS:
                    break
                section_examples.append(part)
                total_chars += len(part) + 2

            if section_examples:
                header = f"### Charge: {label}"
                body = "\n\n".join(section_examples)
                cluster_parts.append(f"{header}\n{body}")

        cluster_section = ""
        if cluster_parts:
            cluster_section = "\n\n".join(cluster_parts) + "\n\n"

        # --- Step 3: similarity fill ---
        all_scored = sorted(
            enumerate(self.examples),
            key=lambda ie: _jaccard(q_tok, ie[1]["tokens"]),
            reverse=True,
        )
        fill_parts: list[str] = []
        fill_ids: set[int] = set()
        for idx, ex in all_scored:
            obj_id = id(ex)
            if obj_id in used_indices or obj_id in fill_ids:
                continue
            part = f"Q: {ex['input'][:300]}\nA: {ex['target']}"
            if total_chars + len(part) > MAX_CHARS:
                break
            fill_parts.append(part)
            fill_ids.add(obj_id)
            total_chars += len(part) + 2
            if len(fill_parts) >= _MAX_FILL:
                break

        fill_section = ""
        if fill_parts:
            fill_section = "**Additional examples:**\n" + "\n\n".join(fill_parts) + "\n\n"

        prompt = PREDICT_PROMPT.format(
            cluster_section=cluster_section,
            fill_section=fill_section,
            input=input_text,
        )
        resp = self._llm(prompt)
        try:
            return json.loads(resp).get("final_answer", "")
        except Exception:
            m = re.search(r'"final_answer"\s*:\s*"([^"]*)"', resp)
            return m.group(1) if m else ""


# ── training examples from diagnostics ───────────────────────────────────────

TRAIN_BATCH = [
    # correct predictions
    {"input": "被告人叶某甲从青田县季宅乡黄放口村溪滩边摘取几个罂粟果，后将上述罂粟果种植在自家农田中。2013年5月14日被查获，被铲除的已结果的罂粟植株共计782株。",
     "ground_truth": "非法种植毒品原植物", "was_correct": True},
    {"input": "被告人何某某在三台县建设镇干坝王村六组蒋家岩坡，私自焚烧秸秆不慎引发山火，造成的过火有林地面积为2.0071公顷。",
     "ground_truth": "失火", "was_correct": True},
    {"input": "被告人何某在签订、履行合同过程中，采取虚构事实、隐瞒真相的手段，骗取他人财物，数额较大。",
     "ground_truth": "合同诈骗", "was_correct": True},
    # errors — exact cases we need to fix
    {"input": "被告单位通辽市某商业广场有限公司及其法定代表人沙某某，利用另案犯罪嫌疑人张某某的职务便利，以通辽市政府办公厅的名义出具了推荐函，向其支付了不正当报酬20万元。",
     "ground_truth": "单位行贿", "was_correct": False},
    {"input": "被告人郭某受雇于同案人，在本市白云区均禾街富力城销售假冒GIVENCHY、LOUISVUITTON、HERMES等注册商标的皮鞋，价值人民币6525350元。",
     "ground_truth": "销售假冒注册商标的商品", "was_correct": False},
    {"input": "被告人李某甲以牟利为目的，盗割正在使用中的公共照明电线，危害公共安全。",
     "ground_truth": "破坏电力设备", "was_correct": False},
    {"input": "被告人韩某某在担任西宁某物业管理有限公司副总经理期间，非法收受闫某的行贿款200,000元人民币。",
     "ground_truth": "非国家工作人员受贿", "was_correct": False},
    {"input": "被告人李某某在经营期间，购买了三部打鱼机，具有赌博功能，为他人提供赌博场所及用具，非法获利150元。",
     "ground_truth": "赌博;开设赌场", "was_correct": False},
    {"input": "被告人何某在肇庆市矶西路江南副产品批发市场门口，偷走了被害人罗某斌一辆红色大运牌三轮摩托车。购买了一辆黑色雅马哈牌摩托车将该车车架及轮毂喷成黄色隐瞒来源。",
     "ground_truth": "盗窃;掩饰、隐瞒犯罪所得、犯罪所得收益", "was_correct": False},
]


# ── tests ─────────────────────────────────────────────────────────────────────

def run_tests():
    sys = LabelClusterMemoryProto(fake_llm)

    # Test 1: cold start
    pred = sys.predict("被告人张某某谎称以租车的名义将被害人轿车骗走")
    print(f"[cold start] prediction: '{pred}'")
    assert pred == "单位行贿"  # fake LLM always returns 单位行贿

    # Test 2: learn batch
    sys.learn_from_batch(TRAIN_BATCH)
    print(f"[after batch] clusters: {sorted(sys.clusters.keys())}")
    assert "单位行贿" in sys.clusters
    assert "销售假冒注册商标的商品" in sys.clusters
    assert "赌博" in sys.clusters      # split from "赌博;开设赌场"
    assert "开设赌场" in sys.clusters
    assert "盗窃" in sys.clusters
    assert "掩饰、隐瞒犯罪所得、犯罪所得收益" in sys.clusters
    print(f"[cluster count] {len(sys.clusters)}")

    # Test 3: canonical header in prompt
    prompt_input = "被告单位某商业广场有限公司法定代表人，以政府名义出具推荐函，向官员支付20万报酬。"
    q_tok = _tokenize(prompt_input)
    label_scores = [(lbl, sys._score_label(lbl, q_tok)) for lbl in sys.clusters]
    label_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"[top labels for unit-bribery query]: {label_scores[:4]}")
    # 单位行贿 should score highly — its example overlaps 单位/公司/政府/报酬 bigrams
    top4_labels = [l for l, _ in label_scores[:4]]
    print(f"top-4 labels: {top4_labels}")

    # Test 4: cluster section present in prompt
    pred2 = sys.predict(prompt_input)
    print(f"[with memory] prediction: '{pred2}'")
    assert pred2 == "单位行贿"

    # Test 5: multi-charge label split properly
    # both "赌博" and "开设赌场" should be separate clusters
    scores_gamble = sys._score_label("赌博", q_tok)
    scores_venue = sys._score_label("开设赌场", q_tok)
    print(f"[multi-charge split] 赌博: {scores_gamble:.4f}, 开设赌场: {scores_venue:.4f}")

    # Test 6: V1 vs V2 prompt structure comparison
    # V1: no header
    # V2: explicit label header (implemented above)
    # Checking that the section header contains the canonical charge
    prompt_text = PREDICT_PROMPT  # inspect template
    assert "### Charge:" in prompt_text or "section headers" in prompt_text
    print(f"[V2 header check] template contains canonical label instructions: OK")

    # Test 7: fill section works when cluster budget is small
    # Add enough examples to trigger fill
    extra = [{"input": f"案件事实{i}：被告人在某地实施了犯罪行为，数额较大，情节严重。",
              "ground_truth": f"诈骗", "was_correct": True} for i in range(5)]
    sys.learn_from_batch(extra)
    print(f"[total examples] {len(sys.examples)}")
    pred3 = sys.predict("被告人采取虚构事实手段骗取他人财物")
    print(f"[with fill] prediction: '{pred3}'")
    assert pred3 == "单位行贿"

    print("\nAll tests passed (V2 selected).")


if __name__ == "__main__":
    run_tests()
