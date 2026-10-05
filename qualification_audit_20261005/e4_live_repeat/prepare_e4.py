"""Prepare E4 without importing or executing project modules.

This independently renders the frozen R4B prompts, verifies the first 20 byte
for byte against the existing R4B score trace, then locks inputs and request
configuration before any GPT-5.6-Sol request is sent.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ENGINE = ROOT / 'lawbench_rsi_v3'
AGENT = ENGINE / 'engine/text_classification/agents/confusion_disambiguation_memory.py'
MEMORY = ROOT / 'reference_examples/text_classification/logs/20260925_200856/LawBench/confusion_disambiguation_memory/gpt-oss-120b/memory.json'
MANIFEST = ENGINE / 'external/prepared_data_v3/manifest_v3.json'
SCORE = ENGINE / 'external/prepared_data_v3/score.jsonl'
REFERENCE_TRACE = ENGINE / 'external/noise/R4B/1/score/score_traces.jsonl'
ISOLATED_CWD = OUT / 'isolated_cwd'
EXPECTED = {
    'agent': '29cfaead2ebda91a803daee6f2460e53707fc5b804fe3d2135859d350cd4ee8b',
    'memory': '841af6631f324cf3297eb12826316ee73b484c329b6484ddba8df3c94c805f79',
    'manifest': '486493d2b3b6241ff92aeca5d18e634b77fd48412ce6fdad1452406f65e1532d',
    'score_order': 'c990c355f90ff59f4b9cf6a44ec157168f1afd3c96ef508c27266d412ddc966c',
    'prompt_template': 'ac0d4dbd23f200ce37f5692640ca766111df27f5109d1b260e571d4e32a032a7',
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_text(text):
    return sha_bytes(text.encode('utf-8'))


def object_hash(value):
    return sha_text(json.dumps(value, sort_keys=True, ensure_ascii=False))


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def tokenize(text):
    return frozenset(re.findall(r'[A-Za-z0-9]+', text.lower()))


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def restore_memory(raw):
    examples = []
    for source in raw.get('examples', []):
        item = dict(source)
        if isinstance(item.get('tokens'), list):
            item['tokens'] = frozenset(item['tokens'])
        else:
            item['tokens'] = tokenize(item.get('raw_question', item.get('input', '')))
        examples.append(item)
    return examples, raw.get('confusion', {})


def build_parts(query, examples, confusion):
    ranked = [(jaccard(tokenize(query), ex['tokens']), index, ex)
              for index, ex in enumerate(examples)]
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top_labels = [ex['target'] for _, _, ex in ranked[:3]]
    top_set, counts = set(top_labels), {}
    for label in top_labels:
        for actual, count in confusion.get(label, {}).items():
            if actual not in top_set:
                counts[actual] = counts.get(actual, 0) + count
    confused = [label for label, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    parts, used, total = [], set(), 0

    def add(index, example):
        nonlocal total
        if index in used:
            return False
        question = example.get('raw_question', example['input'])
        part = f"Q: {question}\nA: {example['target']}"
        if total + len(part) + 2 > 30000:
            return False
        parts.append(part)
        used.add(index)
        total += len(part) + 2
        return True

    if confused:
        needed = set(confused[:3])
        pools = {label: [] for label in needed}
        for score, index, example in ranked:
            if example['target'] in needed:
                pools[example['target']].append((score, index, example))
        for _ in range(2):
            for label in confused[:3]:
                for _, index, example in pools.get(label, []):
                    if index not in used:
                        add(index, example)
                        break
    for _, index, example in ranked:
        if total >= 30000:
            break
        add(index, example)
    return parts


def main():
    if (OUT / 'lock_manifest.json').exists():
        raise SystemExit('E4 inputs are already locked; refusing to overwrite.')
    OUT.mkdir(parents=True, exist_ok=True)
    ISOLATED_CWD.mkdir(exist_ok=True)
    (ISOLATED_CWD / 'README.txt').write_text(
        'Empty isolated working directory for E4 ephemeral Codex turns. No project files are required by the task model.\n',
        encoding='utf-8')
    files = {'agent': AGENT, 'memory': MEMORY, 'manifest': MANIFEST, 'score': SCORE,
             'reference_trace': REFERENCE_TRACE}
    hashes = {key: sha_bytes(path.read_bytes()) for key, path in files.items() if key != 'reference_trace'}
    for key in ('agent', 'memory', 'manifest'):
        if hashes[key] != EXPECTED[key]:
            raise ValueError(f'{key} changed: {hashes[key]}')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if hashes['score'] != manifest['splits']['score']['sha256']:
        raise ValueError('Score split byte hash does not match manifest.')
    score_rows = [json.loads(line) for line in SCORE.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [row['item_id'] for row in score_rows]
    if ids != manifest['splits']['score']['item_ids'] or object_hash(ids) != EXPECTED['score_order']:
        raise ValueError('Score order is not the frozen manifest order.')
    source = AGENT.read_text(encoding='utf-8')
    tree = ast.parse(source)
    template = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROMPT_TEMPLATE' for t in node.targets):
            template = ast.literal_eval(node.value)
    if not isinstance(template, str) or sha_text(template) != EXPECTED['prompt_template']:
        raise ValueError('Frozen prompt template differs.')
    loader = (ENGINE / 'engine/text_classification/data/loaders.py').read_text(encoding='utf-8')
    loader_tree = ast.parse(loader)
    wrapper = next(n for n in loader_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'wrap_lawbench_rows')
    prompt_ast = next(n.value for n in ast.walk(wrapper) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in n.targets))
    if not (isinstance(prompt_ast, ast.JoinedStr) and len(prompt_ast.values) == 2
            and isinstance(prompt_ast.values[0], ast.Constant)):
        raise ValueError('Unexpected score input wrapper.')
    input_prefix = prompt_ast.values[0].value
    memory_raw = json.loads(MEMORY.read_text(encoding='utf-8'))
    examples, confusion = restore_memory(memory_raw)
    historical = []
    with REFERENCE_TRACE.open(encoding='utf-8') as handle:
        for _ in range(20):
            historical.append(json.loads(next(handle)))
    item_dir = OUT / 'locked_prompts'
    item_dir.mkdir(exist_ok=True)
    items, errors = [], []
    for index, row in enumerate(score_rows[:20]):
        runtime_input = input_prefix + row['question']
        target = row['answer'].split('罪名:')[-1].strip() if '罪名:' in row['answer'] else row['answer']
        parts = build_parts(runtime_input, examples, confusion)
        prompt = template.format(examples_section='\n\n'.join(parts), input=runtime_input)
        old = historical[index]
        checks = {
            'item_id': old['item_id'] == row['item_id'],
            'runtime_input': old['input'] == runtime_input,
            'target': old['target'] == target,
            'actual_prompt': old['prompt_text'] == prompt,
            'constructed_prompt': old['constructed_prompt_text'] == prompt,
        }
        if not all(checks.values()):
            errors.append({'index': index, 'item_id': row['item_id'], 'checks': checks})
        prompt_path = item_dir / f'{index:02d}_{row["item_id"]}.txt'
        # Preserve the mixed LF/CRLF characters that are part of LawBench text.
        # Path.write_text would translate LF on Windows and change the payload.
        prompt_path.write_bytes(prompt.encode('utf-8'))
        items.append({'index': index, 'item_id': row['item_id'], 'source_index': row['source_index'],
                      'legacy': row['legacy'], 'input_sha256': sha_text(runtime_input),
                      'target': target, 'target_sha256': sha_text(target),
                      'prompt_path': prompt_path.relative_to(OUT).as_posix(),
                      'prompt_sha256': sha_text(prompt), 'prompt_chars': len(prompt),
                      'selected_examples': len(parts), 'historical_prompt_exact_match': all(checks.values())})
    if errors:
        atomic_json(OUT / 'preflight_errors.json', errors)
        raise ValueError('Independent renderer did not match historical prompts.')
    with (OUT / 'locked_items.jsonl').open('w', encoding='utf-8') as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    request = {
        'protocol': 'official Codex CLI app-server stdio using existing ChatGPT login',
        'client_info': {'name': 'meta_harness_e4_live_repeat',
                        'title': 'Meta-Harness E4 frozen repeat', 'version': '1.0.0'},
        'thread_start': {'model': 'gpt-5.6-sol', 'modelProvider': None,
                         'baseInstructions': 'Reasoning: medium', 'developerInstructions': '',
                         'ephemeral': True, 'cwd': str(ISOLATED_CWD.resolve()),
                         'approvalPolicy': 'never', 'sandbox': 'read-only',
                         'personality': 'none',
                         'config': {'web_search': 'disabled',
                                    'model_reasoning_effort': 'max',
                                    'model_reasoning_summary': 'none'}},
        'turn_start': {'model': 'gpt-5.6-sol', 'effort': 'max', 'summary': 'none',
                       'approvalPolicy': 'never', 'personality': 'none',
                       'sandboxPolicy': {'type': 'readOnly', 'networkAccess': False},
                       'input_rule': 'one text item whose text bytes equal locked prompt; no wrapper',
                       'output_schema': None},
        'per_observation_reset': 'new ephemeral thread per item; frozen memory is rendered outside model; no conversation carryover',
        'client_result_cache': 'old harness cache bypassed; app-server exposes no response-cache switch; server-side reuse remains unknown',
        'tool_policy': 'web search disabled, read-only sandbox, no approval; any tool item invalidates observation and stops run',
        'semantic_difference': 'Codex app-server is an agent transport and may expose core tool schemas/runtime scaffolding. User message and base instruction match the old harness, but raw-completion equivalence cannot be proven. CLI 0.149.1 turn/start exposes no temperature or max-output-token field, so E4 uses the subscription Codex defaults and does not claim equivalence to the old temperature=0/max_tokens=16384 transport.',
    }
    atomic_json(OUT / 'request_config.json', request)
    lock = {
        'status': 'LOCKED_BEFORE_FIRST_LIVE_REQUEST',
        'selection_rule': 'first 20 score items in manifest execution order; no outcome-based selection',
        'selected_item_ids': ids[:20], 'selected_item_ids_sha256': object_hash(ids[:20]),
        'frozen_hashes': {'agent_sha256': hashes['agent'], 'memory_sha256': hashes['memory'],
                          'manifest_sha256': hashes['manifest'], 'score_sha256': hashes['score'],
                          'full_score_item_order_sha256': object_hash(ids),
                          'prompt_template_sha256': sha_text(template),
                          'twenty_prompt_hashes_sha256': object_hash([x['prompt_sha256'] for x in items])},
        'memory_shape': {'examples': len(examples), 'confusion_source_labels': len(confusion)},
        'prompt_validation': {'historical_stage': 'R4B/1/score',
                              'items_exactly_matching_input_target_actual_and_constructed_prompt': 20},
        'state': {'renderer_mutates_memory': False,
                  'frozen_agent_prediction_direct_self_assignments': 0,
                  'prior_E3_persistent_state_changed': False,
                  'future_reset': 'new ephemeral thread for every observation'},
        'request_config_sha256': sha_bytes((OUT / 'request_config.json').read_bytes()),
        'new_model_requests_so_far': 0,
    }
    atomic_json(OUT / 'lock_manifest.json', lock)
    notes = [
        'E4 implementation preflight',
        'R4B code, trained memory, manifest and score bytes match the frozen audit hashes.',
        'The first 20 manifest-ordered score items were locked before any GPT-5.6-Sol request.',
        'Independent rendering matched all 20 historical R4B runtime inputs, targets, constructed prompts and actual prompts byte for byte.',
        'The official Codex transport can set model, max effort, base/developer instructions, ephemeral threads, read-only sandbox and disabled web search.',
        'It cannot prove the absence of internal Codex runtime scaffolding or server-side response reuse. Core shell tool advertisement has no disable field in CLI 0.149.1 schema; any actual tool item is a protocol failure.',
        'Old gpt-oss scores are provenance checks only and are not pooled with E4.',
    ]
    (OUT / 'implementation_notes.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(json.dumps({'status': lock['status'], 'items': len(items),
                      'first_item': items[0], 'hashes': lock['frozen_hashes'],
                      'semantic_difference': request['semantic_difference']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
