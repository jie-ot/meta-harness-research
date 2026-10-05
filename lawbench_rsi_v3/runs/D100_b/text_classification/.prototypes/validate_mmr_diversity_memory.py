"""Self-contained validation of mmr_diversity_memory mechanism.
No filesystem/network/process imports — all agent logic inlined.
"""

import re
import json

# ---------------------------------------------------------------------------
# Inlined agent logic (mirrors agents/mmr_diversity_memory.py exactly)
# ---------------------------------------------------------------------------

def _tokenize(text):
    cjk_chars = re.findall(r'[一-鿿㐀-䶿]', text)
    bigrams = frozenset(
        cjk_chars[i] + cjk_chars[i+1] for i in range(len(cjk_chars)-1)
    )
    ascii_toks = frozenset(re.findall(r'[A-Za-z0-9]+', text.lower()))
    return bigrams | ascii_toks

def _jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def _extract_json_field(text, field, default=""):
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return str(data.get(field, default))
    except Exception:
        pass
    m = re.findall(rf'"{field}"\s*:\s*"([^"]*)"', text)
    return m[-1] if m else default

_EXAMPLE_BUDGET = 24000
_MMR_LAMBDA = 0.6
_MMR_POOL = 60

def _mmr_select(query_toks, candidates, k, lam):
    if not candidates:
        return []
    relevance = [_jaccard(query_toks, c['tokens']) for c in candidates]
    selected = []
    remaining = list(range(len(candidates)))
    for _ in range(min(k, len(candidates))):
        best_i, best_score = None, float('-inf')
        for i in remaining:
            if selected:
                max_sim = max(
                    _jaccard(candidates[i]['tokens'], candidates[s]['tokens'])
                    for s in selected
                )
            else:
                max_sim = 0.0
            score = lam * relevance[i] - (1.0 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    return selected

PROMPT_WITH_GLOSSARY = (
    'Solve the problem below based on the examples provided.\n\n'
    '{examples_section}\n\n'
    '**Problem:**\n{input}\n\n'
    '**Canonical charge labels seen in training (use exact text from this list):**\n{glossary}\n\n'
    'Instructions: use exact label text, semicolons for multiple.\n'
    '{{"reasoning": "...", "final_answer": "..."}}'
)

PROMPT_NO_GLOSSARY = (
    'Solve the problem below based on the examples provided.\n\n'
    '{examples_section}\n\n'
    '**Problem:**\n{input}\n\n'
    '{{"reasoning": "...", "final_answer": "..."}}'
)

class MMRDiversityMemory:
    def __init__(self, llm):
        self._llm = llm
        self.examples = []
        self.label_glossary = []

    def call_llm(self, prompt):
        return self._llm(prompt)

    def _add_labels(self, gt):
        for lbl in gt.split(';'):
            lbl = lbl.strip()
            if lbl and lbl not in self.label_glossary:
                self.label_glossary.append(lbl)

    def _build_glossary_str(self):
        return '\n'.join(f'- {lbl}' for lbl in sorted(self.label_glossary))

    def _build_parts(self, query):
        if not self.examples:
            return []
        q_tok = _tokenize(query)
        pool_ranked = sorted(
            range(len(self.examples)),
            key=lambda i: _jaccard(q_tok, self.examples[i]['tokens']),
            reverse=True,
        )[:_MMR_POOL]
        pool = [self.examples[i] for i in pool_ranked]
        avg_part_len = 300
        k_estimate = max(1, _EXAMPLE_BUDGET // avg_part_len)
        selected_indices = _mmr_select(q_tok, pool, k=k_estimate, lam=_MMR_LAMBDA)
        parts, total = [], 0
        for idx in selected_indices:
            ex = pool[idx]
            q_text = ex.get('raw_question', ex['input'])
            part = f"Q: {q_text}\nA: {ex['target']}"
            if total + len(part) + 2 > _EXAMPLE_BUDGET:
                break
            parts.append(part)
            total += len(part) + 2
        return parts

    def predict(self, inp):
        parts = self._build_parts(inp)
        examples_section = '\n\n'.join(parts)
        if self.label_glossary:
            prompt = PROMPT_WITH_GLOSSARY.format(
                examples_section=examples_section,
                input=inp,
                glossary=self._build_glossary_str(),
            )
        else:
            prompt = PROMPT_NO_GLOSSARY.format(
                examples_section=examples_section,
                input=inp,
            )
        response = self.call_llm(prompt)
        answer = _extract_json_field(response, 'final_answer')
        return answer, {
            'num_examples': len(self.examples),
            'num_selected': len(parts),
            'num_glossary_labels': len(self.label_glossary),
            'prompt': prompt,
        }

    def learn_from_batch(self, batch):
        for r in batch:
            raw_q = r.get('raw_question', r['input'])
            ex = {'input': r['input'], 'target': r['ground_truth'],
                  'tokens': _tokenize(raw_q)}
            if 'raw_question' in r:
                ex['raw_question'] = r['raw_question']
            self.examples.append(ex)
            self._add_labels(r['ground_truth'])

    def get_state(self):
        ser = [{k: (sorted(v) if isinstance(v, frozenset) else v)
                for k, v in ex.items()} for ex in self.examples]
        return json.dumps({'examples': ser, 'label_glossary': self.label_glossary}, indent=2)

    def set_state(self, state):
        data = json.loads(state)
        self.examples = []
        for ex in data.get('examples', []):
            r = dict(ex)
            r['tokens'] = _tokenize(r.get('raw_question', r.get('input', '')))
            self.examples.append(r)
        self.label_glossary = data.get('label_glossary', [])

# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

def fake_llm(prompt):
    return '{"reasoning": "test", "final_answer": "故意伤害;故意毁坏财物"}'

# ---------------------------------------------------------------------------
# Real examples from training diagnostics
# ---------------------------------------------------------------------------

# Cluster A: assault-only (3 single-charge)
ASSAULT_SINGLE = [
    {'input': '被告人张某某持刀捅刺被害人腹部，造成重伤。', 'target': '故意伤害'},
    {'input': '被告人刘某某拿钢筋击打被害人头部，致重伤二级。', 'target': '故意伤害'},
    {'input': '被告人王某某徒手殴打被害人，致轻伤一级。', 'target': '故意伤害'},
]
# Cluster B: assault + secondary charge (multi-label)
ASSAULT_MULTI = [
    {'input': '宋某持钢管殴打陈某甲重伤六级，并打砸店铺财物2609元。',
     'target': '故意伤害;故意毁坏财物'},
    {'input': '被告人李某某带领十余人阻止施工，持顶托打伤刘B轻伤。',
     'target': '故意伤害;聚众扰乱社会秩序'},
]
# Cluster C: entirely different charge types
DIVERSE = [
    {'input': '被告人黄某某向官员行贿三万元以获取高额拆迁补偿。', 'target': '行贿'},
    {'input': '被告人聂某贩卖盗版光碟3500张以营利。', 'target': '侵犯著作权'},
    {'input': '被告人梅某远程锁定他人苹果设备后索要解锁费用。', 'target': '破坏计算机信息系统'},
]

ALL_EXAMPLES = ASSAULT_SINGLE + ASSAULT_MULTI + DIVERSE

def make_batch(examples):
    return [{'input': e['input'], 'raw_question': e['input'],
             'ground_truth': e['target'], 'prediction': e['target'],
             'was_correct': True} for e in examples]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test():
    print("=== Validating mmr_diversity_memory (inlined) ===\n")
    mem = MMRDiversityMemory(fake_llm)

    # 1. Cold start
    ans, meta = mem.predict('被告人某某持刀伤害被害人并砸毁财物。')
    print(f"Cold start: answer='{ans}'  selected={meta['num_selected']}")
    assert meta['num_selected'] == 0
    assert meta['num_glossary_labels'] == 0
    assert 'Problem' in meta['prompt']
    print("  ✓ cold start OK")

    # 2. Learn
    mem.learn_from_batch(make_batch(ALL_EXAMPLES))
    print(f"\nAfter learn: examples={len(mem.examples)}, glossary={len(mem.label_glossary)}")
    assert len(mem.examples) == len(ALL_EXAMPLES)
    assert '故意伤害' in mem.label_glossary
    assert '行贿' in mem.label_glossary
    print("  ✓ learn_from_batch OK")

    # 3. MMR vs Jaccard diversity comparison
    query = '被告人持刀刺伤被害人并砸毁店内财物折价5000元。'
    q_tok = _tokenize(query)

    # Pure Jaccard top-6
    jaccard_idxs = sorted(
        range(len(mem.examples)),
        key=lambda i: _jaccard(q_tok, mem.examples[i]['tokens']),
        reverse=True,
    )[:6]
    jaccard_targets = [mem.examples[i]['target'] for i in jaccard_idxs]

    # MMR top-6
    pool = mem.examples[:_MMR_POOL]
    mmr_idxs = _mmr_select(q_tok, pool, k=6, lam=_MMR_LAMBDA)
    mmr_targets = [pool[i]['target'] for i in mmr_idxs]

    def charge_set(targets):
        s = set()
        for t in targets:
            s.update(c.strip() for c in t.split(';'))
        return s

    j_charges = charge_set(jaccard_targets)
    m_charges = charge_set(mmr_targets)
    multi_in_j = sum(1 for t in jaccard_targets if ';' in t)
    multi_in_m = sum(1 for t in mmr_targets if ';' in t)

    print(f"\nJaccard top-6: {jaccard_targets}")
    print(f"MMR top-6:     {mmr_targets}")
    print(f"Charge coverage — Jaccard: {len(j_charges)}, MMR: {len(m_charges)}")
    print(f"Multi-label examples — Jaccard: {multi_in_j}, MMR: {multi_in_m}")

    assert len(m_charges) >= len(j_charges), \
        f"MMR should cover at least as many charges as Jaccard: {len(m_charges)} vs {len(j_charges)}"
    print("  ✓ MMR charge coverage >= Jaccard coverage")

    # 4. Warm predict — glossary must appear
    ans2, meta2 = mem.predict(query)
    print(f"\nWarm predict: answer='{ans2}'  selected={meta2['num_selected']}")
    assert meta2['num_selected'] > 0
    assert '**Canonical charge labels' in meta2['prompt'], "Glossary missing"
    assert meta2['num_glossary_labels'] > 0
    print("  ✓ warm predict OK")

    # 5. get_state / set_state round-trip
    state = mem.get_state()
    mem2 = MMRDiversityMemory(fake_llm)
    mem2.set_state(state)
    assert len(mem2.examples) == len(mem.examples)
    assert mem2.label_glossary == mem.label_glossary
    assert isinstance(mem2.examples[0]['tokens'], frozenset), \
        "tokens must be frozenset after set_state"
    ans3, meta3 = mem2.predict(query)
    assert meta3['num_selected'] > 0, "Restored memory must retrieve examples"
    print("\nState round-trip: ✓")

    # 6. Budget enforcement — no example should overflow
    total_chars = sum(
        len(f"Q: {ex.get('raw_question', ex['input'])}\nA: {ex['target']}")
        for ex in [pool[i] for i in mmr_idxs]
    )
    assert total_chars <= _EXAMPLE_BUDGET * 1.1, \
        f"Example budget exceeded: {total_chars} > {_EXAMPLE_BUDGET}"
    print("  ✓ budget enforcement OK")

    print("\n✓ All assertions passed — mmr_diversity_memory mechanism validated")

if __name__ == '__main__':
    test()
