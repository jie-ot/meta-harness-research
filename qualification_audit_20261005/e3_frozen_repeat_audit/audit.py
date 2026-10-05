"""E3 bounded offline audit. Standard library only; never imports project code.

Run once with python -X utf8 -I -S -B. All original inputs are read-only.
Large input files are read and hashed in the same pass, never reread.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'lawbench_rsi_v3'
LIMIT = 160 * 1024 * 1024
if (OUT / 'aborted_full_pass_read_ledger.json').exists():
    raise SystemExit('The bounded detailed pass has stopped. Do not reread large inputs; see summary_audit.py and the completed summary artifacts.')
# Conservative upper bound for earlier, small read-only inspection calls.
# No earlier whole trace/call/cache was read. One trace line was inspected.
PRELIMINARY_RESERVE = json.loads((OUT / 'aborted_pass_read_ledger.json').read_text(
    encoding='utf-8'))['next_preliminary_reserve_bytes']
VERSIONS = {
    'R4B': 'confusion_disambiguation_memory',
    'R7A': 'adaptive_tokenizer_confusion_memory',
    'R12B': 'discriminative_similarity_memory',
}
ENGINE_NAMES = ['llm.py', 'inner_loop.py', 'memory_system.py',
                'pilot_observation.py', 'data/api.py', 'data/loaders.py',
                'data/evaluators.py']
MODEL_KEYS = ['solver', 'temperature', 'max_tokens', 'system_prompt',
              'training_mode', 'training_batch_size', 'training_seed',
              'evaluation_workers', 'max_api_retries']
CAT1 = '匹配且独立执行记录支持'
CAT2 = '匹配但缓存或独立性未知'
CAT3 = '不匹配或资料不全'


def rel(path):
    return path.relative_to(ROOT).as_posix()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def object_hash(value):
    # Match the recorded pilot.state.object_hash serialization, read statically.
    return digest_bytes(json.dumps(value, ensure_ascii=False,
                                   sort_keys=True).encode('utf-8'))


def str_hash(value):
    return digest_bytes(value.encode('utf-8'))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')


def write_csv(name, rows):
    if not rows:
        raise ValueError('No rows for ' + name)
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


allow = {}
excluded = []


def permit(path, purpose, required=True):
    p = path.resolve()
    p.relative_to(ROOT)
    if not p.is_file():
        if required:
            raise FileNotFoundError(rel(p))
        return
    if p in allow:
        return
    st = p.stat()
    allow[p] = {'path': rel(p), 'bytes': st.st_size,
                'mtime_ns': st.st_mtime_ns, 'purpose': purpose}


# Exactly 12 evaluation stages. Only four phase-pairs have full trace reads.
for version in VERSIONS:
    for rep in (1, 2):
        for phase in ('score', 'audit'):
            d = BASE / 'external/noise' / version / str(rep) / phase
            for name in ('result.json', 'complete.json', 'start.json',
                         'attempt_times.jsonl', 'failure.json'):
                permit(d / name, 'stage metadata and resume evidence',
                       required=name in ('result.json', 'complete.json', 'start.json'))
            trace = d / f'{phase}_traces.jsonl'
            if phase == 'audit' or version == 'R4B':
                permit(trace, 'complete item, prompt and execution alignment')
            else:
                excluded.append({'path': rel(trace), 'bytes': trace.stat().st_size,
                                 'reason': '160 MiB budget; summary only'})
            calls = d / 'calls.jsonl'
            excluded.append({'path': rel(calls), 'bytes': calls.stat().st_size,
                             'reason': 'raw API event log not read within budget'})
for version, candidate in VERSIONS.items():
    permit(BASE / 'engine/text_classification/agents' / (candidate + '.py'),
           'frozen agent code and fixed template')
    permit(ROOT / 'reference_examples/text_classification/logs/20260925_200856'
           / 'LawBench' / candidate / 'gpt-oss-120b/memory.json',
           'initial memory byte hash only; never execute or print memory')
for name in ENGINE_NAMES:
    permit(BASE / 'engine/text_classification' / name, 'framework and model kwargs')
for name in ('worker.py', 'run_pilot.py', 'state.py', 'injected.py'):
    permit(BASE / 'pilot' / name, 'static reset/resume/accounting/reuse evidence')
for name in ('noise_summary.csv', 'frozen_prompt_consistency.json'):
    permit(BASE / 'analysis' / name, 'existing analysis cross-check')
permit(BASE / 'pilot_config.json', 'safe configuration keys only')
permit(BASE / 'external/prepared_data_v3/manifest_v3.json', 'ordered split manifest')
for phase in ('score', 'audit'):
    permit(BASE / 'external/prepared_data_v3' / (phase + '.jsonl'),
           'ID and input alignment to recorded split')
for run in ('D0_a', 'D0_b', 'D100_a'):
    permit(BASE / 'external/control' / run / 'baseline_cache_recovery.json',
           'H0 restored-cache origin; no cache files read')
for run in ('D100_injected_a', 'D100_injected_b'):
    permit(BASE / 'external/control' / run / 'reused_h0.json',
           'H0 copies deduplicated; no additional stages admitted')

planned = PRELIMINARY_RESERVE + sum(x['bytes'] for x in allow.values())
if planned > LIMIT:
    raise ValueError(f'Preflight exceeds limit: {planned} > {LIMIT}')
dump('read_plan.json', {'limit_bytes': LIMIT,
                      'preliminary_read_upper_bound_bytes': PRELIMINARY_RESERVE,
                      'planned_total_upper_bound_bytes': planned,
                      'allowlist': list(allow.values()), 'excluded': excluded})

manifest = []
read_paths = set()
bytes_read = 0


def read(path, streaming_rows=False):
    global bytes_read
    p = path.resolve()
    if p not in allow or p in read_paths:
        raise ValueError('Unapproved or repeated input read: ' + rel(p))
    record = allow[p]
    before = p.stat()
    if before.st_size != record['bytes'] or before.st_mtime_ns != record['mtime_ns']:
        raise ValueError('Input changed after preflight: ' + rel(p))
    if PRELIMINARY_RESERVE + bytes_read + before.st_size > LIMIT:
        raise ValueError('Read budget exceeded')
    read_paths.add(p)
    hasher = hashlib.sha256()
    if streaming_rows:
        rows = []
        size = 0
        with p.open('rb') as f:
            for line in f:
                size += len(line)
                hasher.update(line)
                if line.strip():
                    rows.append(compact_item(json.loads(line)))
        data = rows
    else:
        data = p.read_bytes()
        size = len(data)
        hasher.update(data)
    bytes_read += size
    after = p.stat()
    if (size, after.st_size, after.st_mtime_ns) != (before.st_size, before.st_size, before.st_mtime_ns):
        raise ValueError('Input changed during read: ' + rel(p))
    manifest.append({**record, 'sha256': hasher.hexdigest(), 'read_bytes': size,
                     'unchanged_during_read': True})
    return data


def read_json(path):
    return json.loads(read(path))


def compact_item(row):
    # Preserve saved decisions. Do not run evaluator, normalize labels or grade.
    metadata = row.get('metadata') or {}
    norm = row.get('normalized_prediction', row.get('normalized_label'))
    if norm is None:
        norm = metadata.get('normalized_prediction', metadata.get('normalized_label'))
    prompt = row.get('prompt_text')
    pred = row.get('prediction')
    return {
        'item_id': row['item_id'], 'input_hash': str_hash(row['input']),
        'target_hash': str_hash(row.get('target', row.get('output', ''))),
        'was_correct': row.get('was_correct'),
        'prediction_hash': str_hash(pred) if isinstance(pred, str) else None,
        'normalized_prediction_hash': object_hash(norm) if norm is not None else None,
        'actual_prompt_hash': str_hash(prompt) if isinstance(prompt, str) and prompt else None,
        'constructed_prompt_hash': str_hash(row['constructed_prompt_text'])
        if isinstance(row.get('constructed_prompt_text'), str) else None,
        'call_ids': row.get('call_ids', []), 'calls_file': row.get('calls_file'),
        'run_id': row.get('run_id'), 'candidate': row.get('candidate'),
        'phase': row.get('phase'), 'saved_metrics': row.get('metrics'),
    }


raw_small = {}
for path, info in allow.items():
    if info['purpose'] == 'complete item, prompt and execution alignment':
        continue
    raw_small[path] = read(path)


def js(path):
    return json.loads(raw_small[path.resolve()])


hashes = {x['path']: x['sha256'] for x in manifest}
config = js(BASE / 'pilot_config.json')
split_manifest_path = BASE / 'external/prepared_data_v3/manifest_v3.json'
split_manifest = js(split_manifest_path)
loader_tree = ast.parse(raw_small[(BASE / 'engine/text_classification/data/loaders.py').resolve()].decode('utf-8-sig'))
wrapper = next(n for n in loader_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'wrap_lawbench_rows')
wrapped_prompt = next(n.value for n in ast.walk(wrapper)
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in n.targets))
if not (isinstance(wrapped_prompt, ast.JoinedStr) and len(wrapped_prompt.values) == 2
        and isinstance(wrapped_prompt.values[0], ast.Constant)
        and isinstance(wrapped_prompt.values[1], ast.FormattedValue)):
    raise ValueError('Unexpected static question wrapper; do not execute it')
# Read a literal prefix from the AST, concatenate saved question text only.
# No project function is executed and no prediction is regraded.
question_prefix = wrapped_prompt.values[0].value
panel = {}
for phase in ('score', 'audit'):
    p = BASE / 'external/prepared_data_v3' / (phase + '.jsonl')
    rows = [json.loads(line) for line in raw_small[p.resolve()].splitlines() if line.strip()]
    panel[phase] = {'ids': [x['item_id'] for x in rows],
                    'inputs': {x['item_id']: str_hash(question_prefix + x['question']) for x in rows},
                    'bytes_hash': hashes[rel(p)],
                    'manifest_payload_hash_matches': hashes[rel(p)] == split_manifest['splits'][phase]['sha256'],
                    'manifest_order_matches': [x['item_id'] for x in rows] == split_manifest['splits'][phase]['item_ids']}

static = {'agents': {}, 'framework': {}, 'source_lines': {},
          'prepared_question_to_input': {'method': 'literal AST prefix plus saved question; no project function execution',
                                         'path': 'lawbench_rsi_v3/engine/text_classification/data/loaders.py',
                                         'line': wrapper.lineno,
                                         'prefix_utf8_sha256': str_hash(question_prefix)},
          'reset': 'Each evaluation process loads an agent, set_state(saved initial memory), and does not learn.',
          'predict_state_caveat': 'get_state equality does not observe every transient field or thread-local state.',
          'cache_claim': 'cache_mode=off, null directory, zero reported hits; raw calls.jsonl not scanned.',
          'api_seed': 'training_seed=42 is not a recorded provider/API seed.',
          'cross_harness_pairing': 'noise() makes independent version-qualified run IDs; no cross-harness repetition block identity established.'}
for version, candidate in VERSIONS.items():
    p = BASE / 'engine/text_classification/agents' / (candidate + '.py')
    src = raw_small[p.resolve()].decode('utf-8-sig')
    tree = ast.parse(src)
    templates = {}
    source_lines = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and 'PROMPT' in target.id:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        templates[target.id] = {'line': node.lineno, 'sha256_utf8': str_hash(node.value.value)}
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ('predict', 'get_state', 'set_state', '_build_parts'):
            source_lines[node.name] = {'start': node.lineno, 'end': node.end_lineno}
    mem = ROOT / 'reference_examples/text_classification/logs/20260925_200856/LawBench' / candidate / 'gpt-oss-120b/memory.json'
    static['agents'][version] = {'candidate': candidate, 'code_path': rel(p), 'code_hash': hashes[rel(p)],
                                 'memory_path': rel(mem), 'memory_hash': hashes[rel(mem)],
                                 'templates': templates, 'function_lines': source_lines}
for name in ENGINE_NAMES:
    p = BASE / 'engine/text_classification' / name
    static['framework'][name] = {'path': rel(p), 'actual_sha256': hashes[rel(p)]}
for name in ('worker.py', 'run_pilot.py', 'state.py', 'injected.py'):
    p = BASE / 'pilot' / name
    tree = ast.parse(raw_small[p.resolve()].decode('utf-8-sig'))
    static['source_lines'][rel(p)] = {node.name: {'start': node.lineno, 'end': node.end_lineno}
                                     for node in tree.body if isinstance(node, ast.FunctionDef)}

reuse = {'restored_caches': [], 'copied_stages': [],
         'policy': 'Restored response-cache entries and reused H0 stage copies do not add independent repetitions. Only the 12 original noise stages are eligible.',
         'copy_bytes_verification': 'Original copied-stage payloads outside the 12-stage allowance not reread; exact source/result digests are declared in the reuse manifests.'}
for run in ('D0_a', 'D0_b', 'D100_a'):
    p = BASE / 'external/control' / run / 'baseline_cache_recovery.json'
    x = js(p)
    counts = {}
    for entry in x['files']:
        counts[entry['phase']] = counts.get(entry['phase'], 0) + 1
    reuse['restored_caches'].append({'run': run, 'path': rel(p), 'source': x['source'],
                                    'created_at_utc': x['created_at_utc'], 'entries_by_phase': counts,
                                    'new_execution_count': 0,
                                    'source_execution_ids': 'missing: not identified by this recovery index'})
for run in ('D100_injected_a', 'D100_injected_b'):
    x = js(BASE / 'external/control' / run / 'reused_h0.json')
    for phase, entry in x['stages'].items():
        reuse['copied_stages'].append({'destination_run': run, 'phase': phase, **entry,
                                       'deduplication_key': entry['source_complete_hash']})

stages = {}
stage_rows = []
issues = []
for phase in panel:
    if not panel[phase]['manifest_payload_hash_matches'] or not panel[phase]['manifest_order_matches']:
        issues.append({'split': phase, 'check': 'raw_split_hash_or_order_disagrees_with_manifest'})
compact_records = {}
for version, candidate in VERSIONS.items():
    for rep in (1, 2):
        for phase in ('score', 'audit'):
            key = f'{version}/{rep}/{phase}'
            d = BASE / 'external/noise' / version / str(rep) / phase
            result = js(d / 'result.json')
            complete = js(d / 'complete.json')
            start = js(d / 'start.json')
            sig = complete['signature']
            usage = result['usage']
            expected_hash = complete['files']['result.json']
            result_verified = hashes[rel(d / 'result.json')] == expected_hash
            start_matches = start['signature'] == sig
            engine_matches = all(hashes[rel(BASE / 'engine/text_classification' / n)] == h
                                 for n, h in sig['engine_hashes'].items())
            code_matches = static['agents'][version]['code_hash'] == sig['code_hash']
            memory_matches = static['agents'][version]['memory_hash'] == sig['memory_hash']
            phase_ids_match = object_hash(panel[phase]['ids']) == sig['item_ids_hash']
            manifest_matches = hashes[rel(split_manifest_path)] == sig['manifest_hash']
            signature_usage_match = usage == complete['usage']
            result_sig_match = all(result.get(k) == sig.get(k) for k in
                                   ('run_id', 'candidate', 'round', 'D', 'phase', 'code_hash', 'memory_hash', 'manifest_hash', 'model_config', 'cache_mode'))
            attempt_path = d / 'attempt_times.jsonl'
            attempts = [json.loads(x) for x in raw_small.get(attempt_path.resolve(), b'').splitlines() if x.strip()]
            failure_path = d / 'failure.json'
            failure = js(failure_path) if failure_path.resolve() in raw_small else None
            trace_path = d / f'{phase}_traces.jsonl'
            rows = None
            trace_matches = None
            if trace_path.resolve() in allow:
                rows = read(trace_path, streaming_rows=True)
                trace_digest = manifest[-1]['sha256']
                trace_matches = trace_digest == complete['files'][trace_path.name]
                compact_records[key] = rows
            checked = {'result_hash_matches_complete': result_verified,
                       'start_signature_matches_complete': start_matches,
                       'result_fields_match_signature': result_sig_match,
                       'result_usage_matches_complete': signature_usage_match,
                       'current_engine_hashes_match': engine_matches,
                       'current_agent_code_hash_matches': code_matches,
                       'initial_memory_hash_matches': memory_matches,
                       'split_item_order_hash_matches': phase_ids_match,
                       'manifest_hash_matches': manifest_matches,
                       'selected_trace_hash_matches_complete': trace_matches}
            for field, ok in checked.items():
                if ok is False:
                    issues.append({'stage': key, 'check': field})
            stages[key] = {'signature': sig, 'result': result, 'complete_at': complete['completed_at_utc'],
                           'start_at': start.get('started_at_utc'), 'attempts': attempts,
                           'failure': failure, 'checks': checked, 'rows': rows}
            stage_rows.append({'stage': key, 'run_id': result['run_id'], 'candidate': candidate,
                               'phase': phase, 'rep': rep, 'correct': result['correct'], 'total': result['total'],
                               'accuracy_pp': 100 * result['accuracy'], 'timestamp': result['timestamp'],
                               'started_at': start.get('started_at_utc'), 'completed_at': complete['completed_at_utc'],
                               'model': result['model'], 'model_config_json': json.dumps(result['model_config'], ensure_ascii=False, sort_keys=True),
                               'code_hash': sig['code_hash'], 'memory_hash': sig['memory_hash'],
                               'item_order_hash': sig['item_ids_hash'], 'manifest_hash': sig['manifest_hash'],
                               'engine_hashes_json': json.dumps(sig['engine_hashes'], sort_keys=True),
                               'cache_mode': sig['cache_mode'], 'cache_directory': sig['cache_dir'],
                               'cache_hits': usage['cache_hits'], 'logical_calls': usage['logical_calls'],
                               'api_requests': usage['api_requests'], 'failed_api_requests': usage['failed_api_requests'],
                               'retry_attempts': usage['retry_attempts'], 'incomplete_attempts': json.dumps(usage['incomplete_attempts']),
                               'nonbillable_credit_rejections': usage['nonbillable_credit_rejections'],
                               'unreported_failed_cost_count': usage['unreported_failed_cost_count'],
                               'cost_usd_reported': usage['reported_usd'], 'cost_usd_estimated': usage['estimated_usd'],
                               'attempt_time_rows': len(attempts), 'retained_failure_status': failure.get('http_status') if failure else None,
                               'persistent_prediction_state_changed': result['persistent_prediction_state_changed'],
                               'full_trace_read': rows is not None, 'raw_calls_read': False,
                               'selected_integrity_checks_json': json.dumps(checked, sort_keys=True)})

existing = list(csv.DictReader(raw_small[(BASE / 'analysis/noise_summary.csv').resolve()].decode('utf-8-sig').splitlines()))
existing_map = {(x['version'], int(x['repetition'])): x for x in existing}
summary = []
qualification = []
aligned = []
comparison_keys = ['candidate', 'phase', 'D', 'manifest_hash', 'item_ids_hash', 'code_hash',
                   'memory_hash', 'model_config', 'cache_mode', 'cache_dir', 'engine_hashes']
for version in VERSIONS:
    for phase in ('score', 'audit'):
        a, b = [stages[f'{version}/{rep}/{phase}'] for rep in (1, 2)]
        protocol_equal = all(a['signature'][k] == b['signature'][k] for k in comparison_keys)
        ra, rb = a['result'], b['result']
        pp1, pp2 = 100 * ra['correct'] / ra['total'], 100 * rb['correct'] / rb['total']
        delta = pp2 - pp1
        for rep, pp in ((1, pp1), (2, pp2)):
            if abs(float(existing_map[(version, rep)][phase + '_pp']) - pp) > 1e-9:
                issues.append({'stage': f'{version}/{rep}/{phase}', 'check': 'summary_index_disagrees'})
        data = None
        reasons = []
        category = CAT3
        if a['rows'] is not None and b['rows'] is not None:
            left, right = a['rows'], b['rows']
            li, ri = [x['item_id'] for x in left], [x['item_id'] for x in right]
            if len(li) != len(set(li)) or len(ri) != len(set(ri)):
                raise ValueError('Duplicate item IDs in ' + version + '/' + phase)
            bm = {x['item_id']: x for x in right}
            ids_equal = li == ri == panel[phase]['ids']
            inputs_equal = ids_equal and all(x['input_hash'] == bm[x['item_id']]['input_hash'] == panel[phase]['inputs'][x['item_id']] for x in left)
            targets_equal = ids_equal and all(x['target_hash'] == bm[x['item_id']]['target_hash'] for x in left)
            counts = dict(CC=0, CW=0, WC=0, WW=0)
            prompts_equal, prompts_known, raw_flips, normalized_known, normalized_flips = 0, 0, 0, 0, 0
            call_sets = []
            for rep_rows in (left, right):
                call_sets.append({c for x in rep_rows for c in x['call_ids']})
            each_one_call = all(len(x['call_ids']) == 1 for x in left + right)
            trace_identity = all(x['run_id'] == stage['result']['run_id'] and x['candidate'] == VERSIONS[version] and x['phase'] == phase
                                 for stage in (a, b) for x in stage['rows'])
            if ids_equal and inputs_equal and targets_equal:
                for x in left:
                    y = bm[x['item_id']]
                    if type(x['was_correct']) is not bool or type(y['was_correct']) is not bool:
                        raise ValueError('Missing saved correctness decision')
                    transition = ('C' if x['was_correct'] else 'W') + ('C' if y['was_correct'] else 'W')
                    counts[transition] += 1
                    known_prompt = bool(x['actual_prompt_hash'] and y['actual_prompt_hash'])
                    equal_prompt = known_prompt and x['actual_prompt_hash'] == y['actual_prompt_hash']
                    prompts_known += int(known_prompt)
                    prompts_equal += int(equal_prompt)
                    raw_flip = x['prediction_hash'] != y['prediction_hash']
                    raw_flips += int(raw_flip)
                    known_norm = x['normalized_prediction_hash'] is not None and y['normalized_prediction_hash'] is not None
                    normalized_known += int(known_norm)
                    if known_norm:
                        normalized_flips += int(x['normalized_prediction_hash'] != y['normalized_prediction_hash'])
                    aligned.append({'version': version, 'phase': phase, 'item_id': x['item_id'],
                                    'input_sha256': x['input_hash'], 'target_sha256': x['target_hash'],
                                    'rep1_saved_correct': x['was_correct'], 'rep2_saved_correct': y['was_correct'],
                                    'transition': transition, 'rep1_prediction_sha256': x['prediction_hash'],
                                    'rep2_prediction_sha256': y['prediction_hash'], 'raw_prediction_text_changed': raw_flip,
                                    'rep1_actual_prompt_sha256': x['actual_prompt_hash'], 'rep2_actual_prompt_sha256': y['actual_prompt_hash'],
                                    'actual_prompt_equal': equal_prompt if known_prompt else None,
                                    'rep1_call_ids': json.dumps(x['call_ids']), 'rep2_call_ids': json.dumps(y['call_ids']),
                                    'normalized_label_comparison': 'missing' if not known_norm else x['normalized_prediction_hash'] == y['normalized_prediction_hash']})
                if sum(counts.values()) != 100 or counts['CC'] + counts['CW'] != ra['correct'] or counts['CC'] + counts['WC'] != rb['correct']:
                    raise ValueError('Saved decisions disagree with stage summaries')
                if abs(100 * (counts['WC'] - counts['CW']) / 100 - delta) > 1e-9:
                    raise ValueError('Net transition identity failed')
            else:
                counts = {k: None for k in counts}
            actual_all_equal = prompts_known == 100 and prompts_equal == 100
            distinct_ids = bool(call_sets[0] and call_sets[1] and not (call_sets[0] & call_sets[1]))
            detailed_integrity = all(v is not False for s in (a, b) for v in s['checks'].values())
            independent_record_support = trace_identity and distinct_ids and each_one_call and all(
                s['result']['usage']['api_requests'] >= 100 and s['signature']['cache_mode'] == 'off'
                and s['signature']['cache_dir'] is None and s['result']['usage']['cache_hits'] == 0
                and s['result']['usage']['incomplete_attempts'] == [] for s in (a, b))
            if protocol_equal and detailed_integrity and ids_equal and inputs_equal and targets_equal and actual_all_equal:
                category = CAT1 if independent_record_support else CAT2
            else:
                if not actual_all_equal:
                    reasons.append(f'实际提示词仅 {prompts_equal}/{prompts_known} 已核对相同')
                if not detailed_integrity:
                    reasons.append('历史签名与现存源文件或所选日志哈希不一致')
                if not (ids_equal and inputs_equal and targets_equal):
                    reasons.append('逐题ID/输入/target不一致')
            if phase == 'audit' and inputs_equal:
                declared = existing_map[(version, 1)]
                if int(declared['wrong_to_correct']) != counts['WC'] or int(declared['correct_to_wrong']) != counts['CW']:
                    issues.append({'pair': version + '/' + phase, 'check': 'existing_audit_transition_index_disagrees'})
            data = {'item_ids_order_equal': ids_equal, 'inputs_equal_and_match_split': inputs_equal,
                    'targets_equal': targets_equal, 'actual_prompts_known': prompts_known,
                    'actual_prompts_equal': prompts_equal, 'trace_call_ids_disjoint': distinct_ids,
                    'one_successful_logical_call_id_per_item': each_one_call,
                    'recorded_execution_identity_matches': trace_identity,
                    'independent_execution_record_support': independent_record_support,
                    'CC': counts['CC'], 'CW': counts['CW'], 'WC': counts['WC'], 'WW': counts['WW'],
                    'raw_prediction_text_flips': raw_flips if inputs_equal else None,
                    'normalized_labels_available_pairs': normalized_known,
                    'normalized_label_flips': normalized_flips if normalized_known == 100 else None}
        else:
            reasons.append('读取预算保留汇总；逐题输入、实际提示词及执行call ID均missing')
        if not protocol_equal:
            category = CAT3
            reasons.append('冻结签名不匹配')
        if any(s['result']['persistent_prediction_state_changed'] is not False for s in (a, b)):
            category = CAT3
            reasons.append('predict持久状态变化；不能归为固定提示词波动')
        if not reasons:
            reasons.append('不同阶段/run ID和不重叠call ID、cache off支持分别执行；不证明统计IID或上游未复用')
        qualification.append({'version': version, 'phase': phase, 'category': category,
                              'protocol_signature_equal': protocol_equal,
                              'initial_state_equal': a['signature']['memory_hash'] == b['signature']['memory_hash'],
                              'predict_saved_state_changed_rep1': ra['persistent_prediction_state_changed'],
                              'predict_saved_state_changed_rep2': rb['persistent_prediction_state_changed'],
                              'full_per_item_alignment': data is not None,
                              'actual_prompts_known': data['actual_prompts_known'] if data else 'missing',
                              'actual_prompts_equal': data['actual_prompts_equal'] if data else 'missing',
                              'trace_call_ids_disjoint': data['trace_call_ids_disjoint'] if data else 'missing',
                              'provider_response_ID_check': 'missing: raw calls.jsonl not scanned',
                              'provider_API_seed': 'missing', 'upstream_backend_fingerprint': 'missing',
                              'raw_API_retry_prompt_check': 'missing: raw calls.jsonl not scanned',
                              'notes': '；'.join(reasons)})
        summary.append({'version': version, 'phase': phase, 'n_repetitions': 2,
                        'items_per_repetition': 100, 'accuracy_rep1_pp': pp1,
                        'accuracy_rep2_pp': pp2, 'delta_rep2_minus_rep1_pp': delta,
                        'range_min_pp': min(pp1, pp2), 'range_max_pp': max(pp1, pp2),
                        'CC': data['CC'] if data else 'missing', 'CW': data['CW'] if data else 'missing',
                        'WC': data['WC'] if data else 'missing', 'WW': data['WW'] if data else 'missing',
                        'raw_prediction_text_flips': data['raw_prediction_text_flips'] if data else 'missing',
                        'normalized_label_flips': data['normalized_label_flips'] if data and data['normalized_label_flips'] is not None else 'missing',
                        'abs_delta_ge_1pp': abs(delta) >= 1, 'abs_delta_ge_2pp': abs(delta) >= 2,
                        'abs_delta_ge_5pp': abs(delta) >= 5, 'qualification': category,
                        'detail_source': 'complete trace pair verified' if data else 'result.json summaries only',
                        'interpretation': 'historically exposed audit; descriptive only' if phase == 'audit' else 'score; descriptive only'})

stat_recheck = []
for p in read_paths:
    st = p.stat()
    old = allow[p]
    ok = (st.st_size, st.st_mtime_ns) == (old['bytes'], old['mtime_ns'])
    stat_recheck.append({'path': rel(p), 'size_mtime_unchanged_after_audit': ok})
    if not ok:
        issues.append({'path': rel(p), 'check': 'source_stat_changed'})

write_csv('repeat_summary.csv', summary)
write_csv('qualification.csv', qualification)
write_csv('stages.csv', stage_rows)
write_csv('aligned_items.csv', aligned)
dump('compact_item_records.json', compact_records)
dump('static_evidence.json', static)
dump('h0_deduplication.json', reuse)
dump('input_manifest.json', manifest)
dump('source_stat_recheck.json', stat_recheck)
dump('read_budget.json', {'limit_bytes': LIMIT, 'preliminary_read_upper_bound_bytes': PRELIMINARY_RESERVE,
                         'new_original_read_bytes': bytes_read,
                         'total_original_read_upper_bound_bytes': PRELIMINARY_RESERVE + bytes_read,
                         'full_trace_files_read': len(compact_records), 'original_stages_admitted': 12,
                         'raw_API_call_files_read': 0, 'cache_files_read': 0,
                         'planned_original_files': len(allow), 'original_files_read': len(read_paths),
                         'hashing_same_pass_as_read': True,
                         'integrity_recheck_strategy': 'selected payload hashes checked against complete.json once; final size/mtime check only, no second large-file read'})
dump('integrity_checks.json', {'status': 'PASS' if not issues else 'ISSUES', 'issues': issues,
                              'pairs': len(summary), 'stage_summaries': len(stage_rows),
                              'aligned_item_pairs': len(aligned), 'new_experimental_model_calls': 0,
                              'forbidden_operations_executed': [],
                              'raw_calls_and_all_completion_hashes_verified': False})
dump('protocol.json', {'purpose': 'E3 frozen-repeat qualification and descriptive consistency; not optimizer-generation variance.',
                       'historical_solver': 'openrouter/openai/gpt-oss-120b',
                       'requested_runtime': 'GPT-6.1 Sol / Max; actual runtime metadata not independently confirmable',
                       'usage': 'No additional quota lookup; no reset, purchase or nested model call.',
                       'safe_configuration': {k: config[k] for k in MODEL_KEYS},
                       'fixed_stage_allowance': 12, 'detailed_phase_pairs': ['R4B/score', 'R4B/audit', 'R7A/audit', 'R12B/audit'],
                       'category_definitions': {CAT1: 'Frozen signatures and actual prompts match, plus distinct execution identities and no logged response-cache use; not proof of IID.',
                                                CAT2: 'Matching protocol/prompt evidence but cache or separate execution unestablished.',
                                                CAT3: 'Prompt/config mismatch or required item/execution evidence incomplete.'},
                       'accuracy_source': 'saved correct/total only, no new evaluation or label normalization',
                       'metrics_prohibited': ['variance estimate', 'confidence interval', 'bootstrap', 'significance', 'probability calibration', 'threshold fitting'],
                       'ranking_flip_analysis': 'omitted: no original matched cross-harness replicate block proven',
                       'audit_status': 'legacy test already exposed; no fresh held-out generalization claim',
                       'threshold_flags': '1/2/5 percentage points descriptive only; at 100 items these are 1/2/5 net cases, with cancellations possible',
                       'next_step': 'stop; no model experiment authorized by this audit'})

lines = [
    'E3：既有冻结版本复测的离线资格审计',
    '目的：核验相同配置下的历史两次评价可比性；不能替代优化器/候选生成方差实验。',
    '设计：3版本×score/audit×2重复，最多12个原阶段；4对完整逐题，2对仅汇总。原文件只读；新增实验模型调用0。',
    '模型：历史均为openrouter/openai/gpt-oss-120b，temperature=0，Reasoning: medium。不得标为SolMax；当前运行模型/effort无可独立确认元数据。',
    f'原始材料累计读取上界：{PRELIMINARY_RESERVE + bytes_read:,} / {LIMIT:,} 字节，含前序读取、中止解析与额外小型核对的保守台账；8个轨迹单次读取同时哈希；不读原始calls或缓存。',
    '',
    '结果（pp为百分点；CW=正确→错误，WC=错误→正确）：',
]
for row in summary:
    q = next(x for x in qualification if x['version'] == row['version'] and x['phase'] == row['phase'])
    lines.append(f"{row['version']} {row['phase']}: {row['accuracy_rep1_pp']:g}→{row['accuracy_rep2_pp']:g}, Δ={row['delta_rep2_minus_rep1_pp']:+g}; "
                 f"CC/CW/WC/WW={row['CC']}/{row['CW']}/{row['WC']}/{row['WW']}; "
                 f"实际提示词相同={q['actual_prompts_equal']}/{q['actual_prompts_known']}; {row['qualification']}。")
lines.extend([
    '', '资格与限制：每组仅2重复；只描述已保存结果，不估计方差、显著性或改善概率。',
    '所有stage初始memory、代码、引擎、模型kwargs、manifest及ID顺序的匹配与文件哈希见stages.csv/static_evidence.json；资格以实际展开提示词为准。',
    'R4B score两次有历史402中断，各18/17次非计费信用拒绝，后恢复；摘要称retry_attempts=0、incomplete_attempts=[]。新逻辑调用118/117，不是118/117个独立评价重复；原calls未扫描，逐attempt完整性未独立核实。',
    '所有result记cache off、cache_hits=0、predict持久state不变；不能证明上游缓存不存在、服务端版本相同或所有瞬时状态不变。',
    'baseline_cache_recovery回填自己H0缓存（D0_a/b各100，D100_a共200），不增加重复；injected的reused_h0复制原score/feedback，也不增加重复，两个injected共用同一D100_a feedback源。',
    '没有持久化规范化预测标签；label flip为missing，原始prediction文本变化不能当成标签翻转。',
    'audit是历史已暴露的old test；不按audit重选版本，不报告新泛化效果。缺少明确跨harness成对重复块证据，不报告排名翻转。',
    '结论：本步仅能支持有资格材料的描述性重复一致性判断；不提供候选生成噪声、SolMax表现或patience阈值校准证据。',
    '下一步影响：E2的prefix回放仍属离线描述；本审计不据此调整delta/patience，也不启动下一实验。',
    f"完整性结果：{'PASS' if not issues else 'ISSUES'}，详见integrity_checks.json；详细读取不足明确missing。",
])
(OUT / '简明中文结论.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PASS' if not issues else 'ISSUES', 'issues': issues,
                  'read_upper_bound_bytes': PRELIMINARY_RESERVE + bytes_read,
                  'limit_bytes': LIMIT,
                  'summary': [{k: x[k] for k in ('version', 'phase', 'accuracy_rep1_pp',
                              'accuracy_rep2_pp', 'delta_rep2_minus_rep1_pp', 'CC', 'CW',
                              'WC', 'WW', 'raw_prediction_text_flips', 'qualification')}
                              for x in summary],
                  'prompt_checks': [{k: x[k] for k in ('version', 'phase',
                                    'actual_prompts_known', 'actual_prompts_equal', 'notes')}
                                    for x in qualification]}, ensure_ascii=False, indent=2))
