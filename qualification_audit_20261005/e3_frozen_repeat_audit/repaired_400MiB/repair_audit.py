"""Authorized E3 repair: fixed 12 stages, offline, standard library only.

No project imports/evaluator execution, no networking, no subprocesses.
Payloads are hashed while streamed. Each stage's compact rows and checkpoint
are persisted before processing the next stage. Scores use saved booleans only.
"""
import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
E3 = OUT.parent
ROOT = E3.parents[1]
BASE = ROOT / 'lawbench_rsi_v3'
LIMIT = 400 * 1024 * 1024
V = {'R4B': 'confusion_disambiguation_memory',
     'R7A': 'adaptive_tokenizer_confusion_memory',
     'R12B': 'discriminative_similarity_memory'}
ENGINE = ['llm.py', 'inner_loop.py', 'memory_system.py', 'pilot_observation.py',
          'data/api.py', 'data/loaders.py', 'data/evaluators.py']
CAT_MATCH = '匹配但缓存或独立性未知'
CAT_BAD = '不匹配或资料不全'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def text_sha(s):
    return sha(s.encode('utf-8')) if isinstance(s, str) else None


def obj_sha(x):
    # Confirmed statically: pilot/state.py:25-26 uses default JSON separators.
    return sha(json.dumps(x, sort_keys=True, ensure_ascii=False).encode('utf-8'))


def rel(p):
    return p.relative_to(ROOT).as_posix()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def csv_save(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def extract_answer(s):
    # Independent text parser implementing the verified historical rule,
    # evaluators.py:12-55. Never calls a project function or model.
    if not s:
        return ''
    text = s.strip()
    if text.startswith('```'):
        payload, in_json = [], False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('```json'):
                in_json = True
                continue
            if stripped.startswith('```') and in_json:
                break
            if in_json:
                payload.append(line)
        if payload:
            text = '\n'.join(payload)
        elif '```' in text:
            parts = text.split('```')
            if len(parts) >= 3:
                text = parts[1]
                if text.startswith('json'):
                    text = text[4:].strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and 'final_answer' in data:
            return str(data['final_answer'])
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and 'final_answer' in data:
                return str(data['final_answer'])
        except json.JSONDecodeError:
            pass
    return s


def canonical(s):
    # Verified evaluators.py:100-111. Separator order is material: use the
    # first present separator, not a new regex split across all separators.
    if not isinstance(s, str):
        return None
    text = extract_answer(s).strip()
    match = re.search(r'\[罪名\](.*?)(?:<eoa>|$)', text)
    if match:
        text = match.group(1).strip()
    elif '罪名:' in text:
        text = text.split('罪名:')[-1]
    text = re.sub(r'<eoa>.*', '', text).strip()
    for sep in [';', '；', ',', '，', '、']:
        if sep in text:
            return sorted({p.strip() for p in text.split(sep) if p.strip()})
    return [text] if text else []


def compact(row):
    errors = []
    for key, typ in (('item_id', str), ('input', str), ('prediction', str),
                     ('target', str), ('was_correct', bool)):
        if type(row.get(key)) is not typ:
            errors.append('required_field_type:' + key)
    pred = canonical(row.get('prediction'))
    target = canonical(row.get('target'))
    return {'item_id': row.get('item_id'), 'input_sha256': text_sha(row.get('input')),
            'target_sha256': text_sha(row.get('target')),
            'was_correct_saved': row.get('was_correct'),
            'prediction_sha256': text_sha(row.get('prediction')),
            'canonical_prediction_derived': pred, 'canonical_target_derived': target,
            'canonical_agrees_saved_correct': ((pred == target) == row['was_correct'])
            if pred is not None and target is not None and type(row.get('was_correct')) is bool else None,
            'actual_prompt_sha256': text_sha(row.get('prompt_text')) if row.get('prompt_text') else None,
            'constructed_prompt_sha256': text_sha(row.get('constructed_prompt_text'))
            if row.get('constructed_prompt_text') else None,
            'call_ids': row.get('call_ids') if isinstance(row.get('call_ids'), list) else None,
            'calls_file': row.get('calls_file'), 'run_id': row.get('run_id'),
            'candidate': row.get('candidate'), 'phase': row.get('phase'),
            'saved_metrics': row.get('metrics'), 'parse_errors': errors}


def selfcheck():
    cases = [('[罪名]甲;乙<eoa>', ['乙', '甲']),
             ('[罪名]甲；乙<eoa>', ['乙', '甲']),
             ('[罪名]甲,乙<eoa>', ['乙', '甲']),
             ('[罪名]甲，乙<eoa>', ['乙', '甲']),
             ('[罪名]甲、乙<eoa>', ['乙', '甲']),
             ('[罪名]甲;乙、丙<eoa>', ['乙、丙', '甲']),
             ('罪名:甲', ['甲']), ('', []),
             ('{"final_answer":"[罪名]甲<eoa>"}', ['甲']),
             ('```json\n{"final_answer":"[罪名]甲<eoa>"}\n```', ['甲']),
             ('prefix {"final_answer":"[罪名]甲<eoa>"} suffix', ['甲'])]
    checks = {f'canonical_case_{i}': canonical(s) == sorted(expected)
              for i, (s, expected) in enumerate(cases)}
    missing = compact({'item_id': 'x', 'input': 'q', 'prediction': '甲',
                       'target': '甲', 'was_correct': True})
    checks['optional_fields_remain_unknown'] = missing['call_ids'] is None and missing['actual_prompt_sha256'] is None
    checks['saved_boolean_preserved'] = missing['was_correct_saved'] is True
    checks['critical_field_error_retained'] = 'required_field_type:was_correct' in compact(
        {'item_id': 'x', 'input': 'q', 'prediction': '甲', 'target': '甲', 'was_correct': 'true'})['parse_errors']
    old = E3 / 'repeat_summary.csv'
    with old.open(encoding='utf-8-sig', newline='') as f:
        recovered = list(csv.DictReader(f))
    checks['six_existing_summary_rows_known'] = len(recovered) == 6
    # One observed original record and one split row, charged within the 2MiB
    # repair preflight reserve. No full trace read in selfcheck.
    p = BASE / 'external/noise/R4B/1/score/score_traces.jsonl'
    with p.open('rb') as f:
        sample = json.loads(f.readline())
    p = BASE / 'external/prepared_data_v3/score.jsonl'
    with p.open('rb') as f:
        split = json.loads(f.readline())
    c = compact(sample)
    checks['actual_schema_prediction_correct_fields'] = not c['parse_errors']
    checks['actual_schema_rendered_and_runtime_separate'] = all(isinstance(sample.get(k), str) for k in
                                                               ('input', 'prompt_text', 'constructed_prompt_text'))
    checks['actual_sample_canonical_agrees_saved'] = c['canonical_agrees_saved_correct'] is True
    checks['prepared_schema_question_not_runtime_input'] = isinstance(split.get('question'), str) and 'input' not in split
    checks['sample_ID_shared'] = sample['item_id'] == split['item_id']
    result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
              'new_model_calls': 0, 'source_rules': 'data/evaluators.py:12-55,100-111; data/loaders.py:167-192',
              'no_project_test_or_evaluator_executed': True}
    save(OUT / 'parser_selfcheck.json', result)
    print(json.dumps({'selfcheck': result['status'], 'checks': len(checks),
                      'failed': [k for k, ok in checks.items() if not ok]}, ensure_ascii=False))
    return all(checks.values())


def main():
    if (OUT / 'execution_started.json').exists():
        raise SystemExit('A bounded repair pass already started. Inspect persisted checkpoints; never automatically reread sources.')
    if not json.loads((OUT / 'parser_selfcheck.json').read_text(encoding='utf-8'))['status'] == 'PASS':
        raise SystemExit('Selfcheck must pass before source streaming.')
    prior = json.loads((E3 / 'read_budget.json').read_text(encoding='utf-8'))['total_original_read_upper_bound_bytes']
    reserve = 2 * 1024 * 1024
    spent = prior + reserve
    allow = {}
    exclusions = []

    def permit(p, purpose, required=True):
        p = p.resolve()
        p.relative_to(ROOT)
        if not p.is_file():
            if required:
                raise FileNotFoundError(rel(p))
            return
        s = p.stat()
        allow[p] = {'path': rel(p), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'purpose': purpose}

    for v, candidate in V.items():
        for rep in (1, 2):
            for phase in ('score', 'audit'):
                d = BASE / 'external/noise' / v / str(rep) / phase
                for n in ('result.json', 'complete.json', 'start.json', 'attempt_times.jsonl', 'failure.json'):
                    permit(d / n, 'stage_metadata', required=n in ('result.json', 'complete.json', 'start.json'))
                permit(d / f'{phase}_traces.jsonl', 'full_trace')
                p = d / 'calls.jsonl'
                exclusions.append({'path': rel(p), 'bytes': p.stat().st_size,
                                   'reason': 'raw API logs excluded; provider identity/cache/attempt qualification remains unknown'})
        permit(BASE / 'engine/text_classification/agents' / (candidate + '.py'), 'agent_code')
        permit(ROOT / 'reference_examples/text_classification/logs/20260925_200856/LawBench'
               / candidate / 'gpt-oss-120b/memory.json', 'initial_memory_hash_only')
    for n in ENGINE:
        permit(BASE / 'engine/text_classification' / n, 'static_framework')
    for n in ('worker.py', 'run_pilot.py', 'state.py', 'injected.py'):
        permit(BASE / 'pilot' / n, 'static_controller')
    for n in ('noise_summary.csv', 'frozen_prompt_consistency.json'):
        permit(BASE / 'analysis' / n, 'small_existing_index')
    mp = BASE / 'external/prepared_data_v3/manifest_v3.json'
    permit(mp, 'manifest')
    for phase in ('score', 'audit'):
        permit(mp.parent / (phase + '.jsonl'), 'prepared_split')
    for run in ('D0_a', 'D0_b', 'D100_a'):
        permit(BASE / 'external/control' / run / 'baseline_cache_recovery.json', 'H0_cache_provenance')
    for run in ('D100_injected_a', 'D100_injected_b'):
        permit(BASE / 'external/control' / run / 'reused_h0.json', 'H0_copy_provenance')
    planned = spent + sum(x['bytes'] for x in allow.values())
    if planned > LIMIT:
        raise SystemExit(f'Fixed allowlist exceeds limit: {planned}>{LIMIT}')
    save(OUT / 'read_plan.json', {'limit_bytes': LIMIT, 'prior_read_upper_bound_bytes': prior,
                                'repair_preflight_and_selfcheck_upper_bound_bytes': reserve,
                                'planned_cumulative_upper_bound_bytes': planned,
                                'allowlist': list(allow.values()), 'excluded': exclusions})
    save(OUT / 'execution_started.json', {'fixed_stages': 12, 'planned_cumulative_upper_bound_bytes': planned,
                                        'previous_parsing_errors_are_execution_deviations': True})
    input_records, read_set, raw, hashes, errors, checkpoints = [], set(), {}, {}, [], []
    original_bytes = 0

    def progress():
        save(OUT / 'progress.json', {'original_bytes_read_this_pass': original_bytes,
                                   'cumulative_original_read_upper_bound_bytes': spent + original_bytes,
                                   'completed_trace_stages': len(checkpoints), 'errors': errors})
        save(OUT / 'input_manifest.json', input_records)

    def begin(p):
        p = p.resolve()
        if p not in allow or p in read_set:
            raise ValueError('Unapproved/repeated read: ' + rel(p))
        s, old = p.stat(), allow[p]
        if (s.st_size, s.st_mtime_ns) != (old['bytes'], old['mtime_ns']):
            raise ValueError('Source changed since allowlist: ' + rel(p))
        if spent + original_bytes + s.st_size > LIMIT:
            raise ValueError('Cumulative read limit')
        read_set.add(p)
        return p, s

    def finish(p, before, h, size):
        nonlocal original_bytes
        original_bytes += size
        now = p.stat()
        same = (size, now.st_size, now.st_mtime_ns) == (before.st_size, before.st_size, before.st_mtime_ns)
        hashes[p] = h.hexdigest()
        input_records.append({**allow[p], 'sha256': h.hexdigest(), 'read_bytes': size,
                              'unchanged_during_read': same})
        if not same:
            errors.append({'source': rel(p), 'problem': 'changed_during_read'})
        progress()

    for p, info in allow.items():
        if info['purpose'] == 'full_trace':
            continue
        p, before = begin(p)
        h, size = hashlib.sha256(), 0
        if info['purpose'] == 'initial_memory_hash_only':
            with p.open('rb') as f:
                while True:
                    b = f.read(1024 * 1024)
                    if not b:
                        break
                    h.update(b)
                    size += len(b)
        else:
            b = p.read_bytes()
            h.update(b)
            size = len(b)
            raw[p] = b
        finish(p, before, h, size)

    def js(p):
        return json.loads(raw[p.resolve()])

    manifest = js(mp)
    loader_p = BASE / 'engine/text_classification/data/loaders.py'
    loader_ast = ast.parse(raw[loader_p.resolve()].decode('utf-8-sig'))
    wrapper = next(n for n in loader_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'wrap_lawbench_rows')
    prompt_ast = next(n.value for n in ast.walk(wrapper) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in n.targets))
    if not (isinstance(prompt_ast, ast.JoinedStr) and len(prompt_ast.values) == 2
            and isinstance(prompt_ast.values[0], ast.Constant)
            and isinstance(prompt_ast.values[1].value, ast.Subscript)
            and isinstance(prompt_ast.values[1].value.slice, ast.Constant)
            and prompt_ast.values[1].value.slice.value == 'question'):
        errors.append({'problem': 'static_question_wrapper_structure_unexpected'})
        progress()
        raise SystemExit('New substantive schema error; saved metadata retained. Do not automatically retry.')
    prefix = prompt_ast.values[0].value
    panels = {}
    for phase in ('score', 'audit'):
        p = mp.parent / (phase + '.jsonl')
        rows = [json.loads(line) for line in raw[p.resolve()].splitlines() if line.strip()]
        ids = [x['item_id'] for x in rows]
        panels[phase] = {'ids': ids, 'inputs': {x['item_id']: text_sha(prefix + x['question']) for x in rows},
                         'targets': {x['item_id']: text_sha(x['answer'].split('罪名:')[-1].strip()
                                     if '罪名:' in x['answer'] else x['answer']) for x in rows},
                         'source_hash_matches_manifest': hashes[p.resolve()] == manifest['splits'][phase]['sha256'],
                         'source_order_matches_manifest': ids == manifest['splits'][phase]['item_ids'],
                         'item_order_hash': obj_sha(ids)}
    existing = list(csv.DictReader(raw[(BASE / 'analysis/noise_summary.csv').resolve()].decode('utf-8-sig').splitlines()))
    if not all({'version', 'repetition', 'score_pp', 'audit_pp'} <= set(x) for x in existing):
        errors.append({'problem': 'noise_summary_header_changed'})
        progress()
        raise SystemExit('New substantive index error; no trace reread. Metadata checkpoints retained.')
    index = {(x['version'], int(x['repetition'])): x for x in existing}
    static = {'agents': {}, 'framework_sha256': {n: hashes[(BASE / 'engine/text_classification' / n).resolve()] for n in ENGINE},
              'schema': {'prepared_question': 'raw split question; wrapper literal prefix added independently',
                         'runtime_input': 'trace.input, checked against wrapper + question',
                         'actual_prompt': 'trace.prompt_text, last captured api_result/cache_hit prompt according to inner_loop.py:454-456',
                         'constructed_prompt': 'trace.constructed_prompt_text, memory-system constructed request before captured actual prompt override',
                         'prediction': 'trace.prediction, extracted answer; not metadata.full_response',
                         'correct': 'trace.was_correct, saved boolean; no new evaluation',
                         'canonical': 'derived from saved prediction using verified evaluator text parsing only; not persisted provider label'},
              'canonical_rule_source': {'path': 'lawbench_rsi_v3/engine/text_classification/data/evaluators.py',
                                        'sha256': hashes[(BASE / 'engine/text_classification/data/evaluators.py').resolve()],
                                        'extract_final_answer_lines': [12, 55], 'parse_charges_lines': [100, 111]},
              'provider_seed': 'unknown; training_seed 42 is not an API seed',
              'upstream_model_version_cache_independence': 'unknown; no raw calls/backend identity verification',
              'reset': 'worker.py:107-125 loads same hashed memory independently and resumes only signature-matched saved items; no learning.',
              'persistent_state': 'result boolean only; does not observe every transient field',
              'cross_harness_repetition_blocks': 'not established; noise() version-qualified run IDs; rankings omitted'}
    for v, candidate in V.items():
        p = BASE / 'engine/text_classification/agents' / (candidate + '.py')
        source = raw[p.resolve()].decode('utf-8-sig')
        tree = ast.parse(source)
        template, functions, mutations = {}, {}, []
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                for t in n.targets:
                    if isinstance(t, ast.Name) and 'PROMPT' in t.id:
                        template[t.id] = {'line': n.lineno, 'sha256': text_sha(n.value.value)}
            if isinstance(n, ast.FunctionDef) and n.name in ('predict', '_build_parts', 'get_state', 'set_state'):
                functions[n.name] = {'start': n.lineno, 'end': n.end_lineno}
                if n.name in ('predict', '_build_parts'):
                    for a in ast.walk(n):
                        targets = a.targets if isinstance(a, ast.Assign) else [a.target] if isinstance(a, (ast.AnnAssign, ast.AugAssign)) else []
                        if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self' for t in targets):
                            mutations.append({'line': a.lineno, 'source': ast.get_source_segment(source, a)})
        mem = ROOT / 'reference_examples/text_classification/logs/20260925_200856/LawBench' / candidate / 'gpt-oss-120b/memory.json'
        static['agents'][v] = {'candidate': candidate, 'code_path': rel(p), 'code_sha256': hashes[p.resolve()],
                               'initial_memory_path': rel(mem), 'initial_memory_sha256': hashes[mem.resolve()],
                               'templates': template, 'functions': functions,
                               'direct_self_assignments_in_prediction_helpers': mutations}
    save(OUT / 'static_evidence.json', static)

    stage_map, item_map, stage_rows = {}, {}, []
    for v, candidate in V.items():
        for rep in (1, 2):
            for phase in ('score', 'audit'):
                key = f'{v}/{rep}/{phase}'
                d = BASE / 'external/noise' / v / str(rep) / phase
                result, complete, start = [js(d / n) for n in ('result.json', 'complete.json', 'start.json')]
                p, before = begin(d / f'{phase}_traces.jsonl')
                compact_path = OUT / 'stage_records' / f'{v}_{rep}_{phase}.jsonl'
                compact_path.parent.mkdir(exist_ok=True)
                h, size, records, stage_errors = hashlib.sha256(), 0, [], []
                with p.open('rb') as f, compact_path.open('w', encoding='utf-8') as target:
                    for line_number, line in enumerate(f, 1):
                        h.update(line)
                        size += len(line)
                        if not line.strip():
                            continue
                        try:
                            row = compact(json.loads(line))
                        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                            row = {'item_id': None, 'parse_errors': ['JSON_or_core_type:' + type(exc).__name__]}
                        row['source_line'] = line_number
                        if row['parse_errors']:
                            stage_errors.append({'line': line_number, 'errors': row['parse_errors']})
                        records.append(row)
                        target.write(json.dumps(row, ensure_ascii=False) + '\n')
                        target.flush()
                finish(p, before, h, size)
                sig, usage = complete['signature'], result['usage']
                ids = [x.get('item_id') for x in records]
                checks = {'trace_hash_matches_complete': hashes[p] == complete['files'][p.name],
                          'result_hash_matches_complete': hashes[(d / 'result.json').resolve()] == complete['files']['result.json'],
                          'start_signature_matches_complete': start['signature'] == sig,
                          'result_fields_match_signature': all(result.get(k) == sig.get(k) for k in
                            ('run_id', 'candidate', 'phase', 'D', 'round', 'code_hash', 'memory_hash', 'manifest_hash', 'model_config', 'cache_mode')),
                          'result_usage_matches_complete': usage == complete['usage'],
                          'current_code_hash_matches': static['agents'][v]['code_sha256'] == sig['code_hash'],
                          'initial_memory_hash_matches': static['agents'][v]['initial_memory_sha256'] == sig['memory_hash'],
                          'engine_hashes_match': static['framework_sha256'] == sig['engine_hashes'],
                          'manifest_hash_matches': hashes[mp.resolve()] == sig['manifest_hash'],
                          'split_payload_and_order_valid': panels[phase]['source_hash_matches_manifest'] and panels[phase]['source_order_matches_manifest'],
                          'item_order_hash_matches': panels[phase]['item_order_hash'] == sig['item_ids_hash'],
                          'trace_item_order_matches_panel': ids == panels[phase]['ids'],
                          'trace_unique_100_items': len(ids) == len(set(ids)) == 100,
                          'trace_input_matches_wrapped_split': all(x.get('input_sha256') == panels[phase]['inputs'].get(x.get('item_id')) for x in records),
                          'trace_target_matches_wrapped_split': all(x.get('target_sha256') == panels[phase]['targets'].get(x.get('item_id')) for x in records),
                          'saved_booleans_sum_matches_result': sum(x.get('was_correct_saved') is True for x in records) == result['correct'] and len(records) == result['total'],
                          'canonical_parser_agrees_all_saved_booleans': all(x.get('canonical_agrees_saved_correct') is True for x in records),
                          'index_accuracy_matches': abs(float(index[(v, rep)][phase + '_pp']) - 100 * result['accuracy']) < 1e-9,
                          'trace_execution_identity_matches': all(x.get('run_id') == result['run_id'] and x.get('candidate') == candidate and x.get('phase') == phase for x in records)}
                ap, fp = d / 'attempt_times.jsonl', d / 'failure.json'
                attempts = [json.loads(x) for x in raw.get(ap.resolve(), b'').splitlines() if x.strip()]
                failure = js(fp) if fp.resolve() in raw else None
                checkpoint = {'stage': key, 'rows': len(records), 'trace_bytes': size, 'trace_sha256': hashes[p],
                              'compact_records': rel(compact_path), 'compact_records_sha256': sha(compact_path.read_bytes()),
                              'checks': checks, 'parse_errors': stage_errors,
                              'current_cumulative_read_upper_bound_bytes': spent + original_bytes}
                save(OUT / 'stage_checkpoints' / f'{v}_{rep}_{phase}.json', checkpoint)
                checkpoints.append(checkpoint)
                item_map[key] = records
                stage_map[key] = {'result': result, 'signature': sig, 'checks': checks}
                stage_rows.append({'stage': key, 'run_id': result['run_id'], 'candidate': candidate,
                                   'phase': phase, 'repetition': rep, 'correct': result['correct'], 'total': result['total'],
                                   'accuracy_pp': 100 * result['accuracy'], 'started_utc': start['started_at_utc'],
                                   'completed_utc': complete['completed_at_utc'], 'model': result['model'],
                                   'model_config_json': json.dumps(sig['model_config'], ensure_ascii=False, sort_keys=True),
                                   'code_sha256': sig['code_hash'], 'memory_sha256': sig['memory_hash'],
                                   'item_order_sha256': sig['item_ids_hash'], 'manifest_sha256': sig['manifest_hash'],
                                   'cache_mode': sig['cache_mode'], 'cache_dir': sig['cache_dir'], 'cache_hits': usage['cache_hits'],
                                   'logical_calls': usage['logical_calls'], 'api_requests': usage['api_requests'],
                                   'failed_api_requests': usage['failed_api_requests'], 'retry_attempts': usage['retry_attempts'],
                                   'nonbillable_credit_rejections': usage['nonbillable_credit_rejections'],
                                   'incomplete_attempts_json': json.dumps(usage['incomplete_attempts']),
                                   'attempt_time_rows': len(attempts), 'retained_failure_HTTP': failure.get('http_status') if failure else None,
                                   'persistent_prediction_state_changed': result['persistent_prediction_state_changed'],
                                   'canonical_prediction_source': 'derived exact historical text parsing, saved correctness unchanged',
                                   'trace_sha256': hashes[p], 'source': rel(p)})
                progress()
                if stage_errors or any(not ok for ok in checks.values()):
                    errors.append({'stage': key, 'stage_errors': stage_errors,
                                   'failed_checks': [k for k, ok in checks.items() if not ok]})
                    progress()
                    raise SystemExit('New substantive source/decoder discrepancy; completed stage records retained. Stop, report, do not reread.')

    summaries, qualifications, aligned = [], [], []
    fields = ['candidate', 'phase', 'D', 'code_hash', 'memory_hash', 'manifest_hash',
              'item_ids_hash', 'model_config', 'cache_mode', 'cache_dir', 'engine_hashes']
    for v in V:
        for phase in ('score', 'audit'):
            a, b = [stage_map[f'{v}/{rep}/{phase}'] for rep in (1, 2)]
            left, right = [item_map[f'{v}/{rep}/{phase}'] for rep in (1, 2)]
            rm = {x['item_id']: x for x in right}
            counts = dict(CC=0, CW=0, WC=0, WW=0)
            raw_flips = label_flips = actual_equal = constructed_equal = 0
            actual_available = constructed_available = 0
            inputs_equal = targets_equal = True
            mismatch_ids = []
            for x in left:
                y = rm[x['item_id']]
                transition = ('C' if x['was_correct_saved'] else 'W') + ('C' if y['was_correct_saved'] else 'W')
                counts[transition] += 1
                inputs_equal &= x['input_sha256'] == y['input_sha256']
                targets_equal &= x['target_sha256'] == y['target_sha256']
                raw_changed = x['prediction_sha256'] != y['prediction_sha256']
                label_changed = x['canonical_prediction_derived'] != y['canonical_prediction_derived']
                raw_flips += int(raw_changed)
                label_flips += int(label_changed)
                actual_known = bool(x['actual_prompt_sha256'] and y['actual_prompt_sha256'])
                constructed_known = bool(x['constructed_prompt_sha256'] and y['constructed_prompt_sha256'])
                actual_match = actual_known and x['actual_prompt_sha256'] == y['actual_prompt_sha256']
                constructed_match = constructed_known and x['constructed_prompt_sha256'] == y['constructed_prompt_sha256']
                actual_available += int(actual_known)
                actual_equal += int(actual_match)
                constructed_available += int(constructed_known)
                constructed_equal += int(constructed_match)
                if actual_known and not actual_match:
                    mismatch_ids.append(x['item_id'])
                aligned.append({'version': v, 'phase': phase, 'item_id': x['item_id'],
                                'input_sha256_rep1': x['input_sha256'], 'input_sha256_rep2': y['input_sha256'],
                                'target_sha256_rep1': x['target_sha256'], 'target_sha256_rep2': y['target_sha256'],
                                'saved_correct_rep1': x['was_correct_saved'], 'saved_correct_rep2': y['was_correct_saved'],
                                'transition': transition, 'raw_text_changed': raw_changed,
                                'canonical_label_set_changed': label_changed,
                                'canonical_rep1_json': json.dumps(x['canonical_prediction_derived'], ensure_ascii=False),
                                'canonical_rep2_json': json.dumps(y['canonical_prediction_derived'], ensure_ascii=False),
                                'actual_prompt_sha256_rep1': x['actual_prompt_sha256'], 'actual_prompt_sha256_rep2': y['actual_prompt_sha256'],
                                'actual_prompt_equal': actual_match, 'constructed_prompt_equal': constructed_match,
                                'call_ids_rep1_json': json.dumps(x['call_ids']), 'call_ids_rep2_json': json.dumps(y['call_ids'])})
            all_calls = [{c for x in rows for c in (x['call_ids'] or [])} for rows in (left, right)]
            call_disjoint = len(all_calls[0]) == len(all_calls[1]) == 100 and not (all_calls[0] & all_calls[1])
            one_call_per_item = all(isinstance(x['call_ids'], list) and len(x['call_ids']) == 1 for x in left + right)
            signatures_equal = all(a['signature'][k] == b['signature'][k] for k in fields)
            r1, r2 = a['result'], b['result']
            p1, p2 = [100 * r['correct'] / r['total'] for r in (r1, r2)]
            delta = p2 - p1
            arithmetic = sum(counts.values()) == 100 and counts['CC'] + counts['CW'] == r1['correct'] and counts['CC'] + counts['WC'] == r2['correct'] and delta == counts['WC'] - counts['CW']
            stable_saved_state = all(s['result']['persistent_prediction_state_changed'] is False for s in (a, b))
            response_cache_off = all(s['signature']['cache_mode'] == 'off' and s['signature']['cache_dir'] is None
                                     and s['result']['usage']['cache_hits'] == 0 for s in (a, b))
            matched = (signatures_equal and inputs_equal and targets_equal and actual_equal == actual_available == 100
                       and one_call_per_item and call_disjoint and stable_saved_state and response_cache_off)
            category = CAT_MATCH if matched else CAT_BAD
            if not arithmetic:
                errors.append({'pair': v + '/' + phase, 'problem': 'transition_identity_failed'})
            old_transitions_match = None
            if phase == 'audit':
                old_transitions_match = counts['CW'] == int(index[(v, 1)]['correct_to_wrong']) and counts['WC'] == int(index[(v, 1)]['wrong_to_correct'])
                if not old_transitions_match:
                    errors.append({'pair': v + '/' + phase, 'problem': 'old_transition_summary_disagrees'})
            summaries.append({'version': v, 'phase': phase, 'n_repetitions': 2, 'items_per_repetition': 100,
                              'accuracy_rep1_pp': p1, 'accuracy_rep2_pp': p2, 'delta_rep2_minus_rep1_pp': delta,
                              'range_min_pp': min(p1, p2), 'range_max_pp': max(p1, p2), **counts,
                              'canonical_label_set_flips': label_flips, 'raw_prediction_text_flips': raw_flips,
                              'actual_prompt_equal_items': actual_equal, 'actual_prompt_compared_items': actual_available,
                              'constructed_prompt_equal_items': constructed_equal,
                              'abs_delta_ge_1pp': abs(delta) >= 1, 'abs_delta_ge_2pp': abs(delta) >= 2,
                              'abs_delta_ge_5pp': abs(delta) >= 5, 'transition_arithmetic_valid': arithmetic,
                              'old_audit_transition_summary_matches': old_transitions_match,
                              'qualification': category})
            qualifications.append({'version': v, 'phase': phase, 'category': category,
                                   'frozen_signature_equal': signatures_equal, 'item_ID_and_order_equal': [x['item_id'] for x in left] == [x['item_id'] for x in right],
                                   'runtime_inputs_equal': inputs_equal, 'targets_equal': targets_equal,
                                   'actual_prompts_equal': actual_equal, 'actual_prompts_known': actual_available,
                                   'constructed_prompts_equal': constructed_equal, 'actual_prompt_mismatch_item_IDs_json': json.dumps(mismatch_ids),
                                   'initial_memory_byte_hash_verified': True, 'agent_and_framework_byte_hash_verified': True,
                                   'persistent_state_changed_rep1': r1['persistent_prediction_state_changed'],
                                   'persistent_state_changed_rep2': r2['persistent_prediction_state_changed'],
                                   'cache_mode_rep1': a['signature']['cache_mode'], 'cache_mode_rep2': b['signature']['cache_mode'],
                                   'cache_hits_rep1': r1['usage']['cache_hits'], 'cache_hits_rep2': r2['usage']['cache_hits'],
                                   'trace_call_IDs_disjoint': call_disjoint, 'one_call_ID_per_item': one_call_per_item,
                                   'separate_execution_records_support': 'distinct version-qualified run IDs, trace hashes, nonoverlapping call IDs; no H0 copy counted',
                                   'API_attempt_event_reconciliation': 'unknown: raw calls not scanned',
                                   'provider_response_ID_backend_API_seed': 'unknown', 'upstream_cache_and_statistical_independence': 'unknown',
                                   'notes': '记录匹配但不证明统计独立/服务端缓存状态；不升级为第一类' if matched else '实际展开prompt或必需记录不匹配；不能归为固定prompt噪声'})

    ids_by_stage = {k: {c for x in rows for c in (x['call_ids'] or [])} for k, rows in item_map.items()}
    global_call_count = sum(len(x) for x in ids_by_stage.values())
    global_unique_calls = len(set().union(*ids_by_stage.values()))
    reuse = {'restored_caches': [], 'copied_stages': [],
             'only_original_noise_stages_counted': 12,
             'policy': 'Restored own-H0 response caches and injected copies are never added as repetitions; copied payloads outside fixed stages not reread.'}
    for run in ('D0_a', 'D0_b', 'D100_a'):
        x = js(BASE / 'external/control' / run / 'baseline_cache_recovery.json')
        count = {}
        for f in x['files']:
            count[f['phase']] = count.get(f['phase'], 0) + 1
        reuse['restored_caches'].append({'run': run, 'declared_source': x['source'], 'entries_by_phase': count,
                                         'additional_repetitions': 0, 'source_execution_ID': 'unknown in recovery index'})
    for run in ('D100_injected_a', 'D100_injected_b'):
        x = js(BASE / 'external/control' / run / 'reused_h0.json')
        for phase, entry in x['stages'].items():
            reuse['copied_stages'].append({'destination': run, 'phase': phase, **entry, 'dedup_key': entry['source_complete_hash']})
    reuse['noise_stage_call_ID_sum'] = global_call_count
    reuse['noise_stage_call_ID_unique'] = global_unique_calls
    reuse['no_successful_call_ID_reused_between_noise_stages'] = global_call_count == global_unique_calls == 1200
    source_stats = []
    for p in read_set:
        s, old = p.stat(), allow[p]
        unchanged = (s.st_size, s.st_mtime_ns) == (old['bytes'], old['mtime_ns'])
        source_stats.append({'path': rel(p), 'unchanged_size_mtime': unchanged})
        if not unchanged:
            errors.append({'path': rel(p), 'problem': 'source_changed_after_read'})
    csv_save(OUT / 'repeat_summary.csv', summaries)
    csv_save(OUT / 'qualification.csv', qualifications)
    csv_save(OUT / 'stages.csv', stage_rows)
    csv_save(OUT / 'aligned_items.csv', aligned)
    save(OUT / 'h0_deduplication.json', reuse)
    save(OUT / 'source_stat_recheck.json', source_stats)
    save(OUT / 'read_budget.json', {'limit_bytes': LIMIT, 'prior_original_read_upper_bound_bytes': prior,
                                 'repair_preflight_and_selfcheck_upper_bound_bytes': reserve,
                                 'new_original_read_bytes': original_bytes,
                                 'total_original_read_upper_bound_bytes': spent + original_bytes,
                                 'full_trace_files_read_this_pass': 12, 'full_trace_files_reread_for_verification': 0,
                                 'raw_calls_files_fully_read': 0, 'cache_files_read': 0,
                                 'hashing_same_stream_as_parse': True, 'stages_admitted': 12})
    save(OUT / 'integrity_checks.json', {'status': 'PASS' if not errors else 'ERRORS', 'errors': errors,
                                       'stage_count': 12, 'paired_rows': len(aligned), 'summary_rows': len(summaries),
                                       'separate_execution_records_distinct_call_IDs': global_unique_calls,
                                       'new_experimental_model_calls': 0,
                                       'scientific_independence': 'unknown; not a PASS criterion'})
    save(OUT / 'execution_deviations.json', {'original_attempts': [
        {'error': 'prepared split question mistaken for runtime input', 'large_traces_read': 0, 'resolution': 'verified loaders AST wrapper; raw question, runtime input, rendered prompt treated separately'},
        {'error': 'noise CSV repetition mistaken for rep', 'large_traces_read': 8, 'resolution': 'verified exact header; all 12 stages re-read within expressly authorized 400MiB cumulative limit'}],
        'original_summary_only_result': 'superseded by repaired_400MiB results; retained as execution history, not research conclusion',
        'repair': 'small parser selfcheck PASS; fixed allowlist; same-stream SHA and incremental per-item/stage records; missing optional fields unknown rather than exceptions',
        'sources_modified': False, 'new_models_or_tests_or_experiments': False,
        'threshold_or_scientific_rules_relaxed_for_PASS': False})
    save(OUT / 'protocol.json', {'scope': 'original 12 frozen stages only, no new inference',
                               'historical_model': 'openrouter/openai/gpt-oss-120b, temperature0, Reasoning medium',
                               'current_runtime_model_effort': 'requested GPT-6.1 Sol / Max; actual metadata not independently confirmable',
                               'canonical_labels': 'independently derived using existing evaluator exact text parsing; scores from saved booleans only',
                               'audit': 'old test historically exposed, no fresh generalization claim, no audit reselection',
                               'ranking_flips': 'omitted: original matched cross-harness repetition blocks not proven',
                               'n': '2 repetitions per phase/version; descriptive counts only',
                               'no_estimators': ['no variance', 'no CI/bootstrap', 'no significance', 'no calibration', 'no threshold fitting'],
                               'classification': {'匹配且独立执行记录支持': 'not assigned; missing full API/provenance qualification',
                                                  CAT_MATCH: 'frozen byte hashes and all actual per-item inputs/prompts match; cache/backend/statistical independence unknown',
                                                  CAT_BAD: 'actual prompt or required data mismatch/incomplete'},
                               'stop': 'finish outputs and stop; zero experimental model calls'})
    lines = ['E3修复后：同一12阶段冻结复测的逐题资格审计',
             '目的：核验既有冻结评价可比性；不替代优化器或候选生成方差实验。',
             '设计：3版本×score/audit×2重复，每次100题；固定原12stage，源文件只读，新实验模型调用0。',
             '历史模型均gpt-oss-120b，temperature=0，Reasoning: medium；不标为SolMax。当前模型/effort不可独立确认。',
             '结果（CC/CW/WC/WW依次正确→正确、正确→错误、错误→正确、错误→错误；canonical flips是集合变化数）：']
    for row in summaries:
        lines.append(f"{row['version']} {row['phase']}: {row['accuracy_rep1_pp']:g}→{row['accuracy_rep2_pp']:g}，Δ={row['delta_rep2_minus_rep1_pp']:+g}pp，范围[{row['range_min_pp']:g},{row['range_max_pp']:g}]；CC/CW/WC/WW={row['CC']}/{row['CW']}/{row['WC']}/{row['WW']}；canonical flips={row['canonical_label_set_flips']}，raw flips={row['raw_prediction_text_flips']}；actual prompt相同{row['actual_prompt_equal_items']}/100；{row['qualification']}。")
    lines.extend(['资格：12份原trace SHA均核对complete，代码、框架、初始memory字节hash、config、split顺序和逐题input/target全部核查；规范化标签按历史解析规则派生并与全部1200条原保存正确性相容，准确率不重新评估。',
                  '各stage cache off/null、cache_hits0、persistent state_changed=false，成功trace共1200个call ID且跨stage不重复；仍不证明服务端缓存、同一backend或统计独立，原API大日志未完整扫描，所以不升为第一类。',
                  '模板固定不代表实际prompt固定；实际prompt差异见qualification.csv/逐题表。若不同，只能称同配置/状态过程的重复差异，不能隔离为固定prompt噪声。',
                  'R4B score原402中断后恢复，摘要18/17非计费拒绝、logical/API118/117，成功记录各100；其余stage各100请求。retry/incomplete仅依据摘要，完整API attempt核对unknown。',
                  'H0恢复缓存、reused_h0副本去重，不增加重复；两个injected反馈共用同一D100_a源。audit是已暴露old test，不据其重选。跨harness原成对重复块缺证，不报告排名翻转。',
                  '阈值1/2/5pp只作描述，100题对应1/2/5净病例，正反翻转可抵消。n=2不估方差/区间/显著性/改善概率，不校准patience。',
                  '解析错误属于执行偏离：已查实question/input与repetition/rep字段，20项小型自检后增量保存再流式修复；初次未完成资格不是研究结论。详情见execution_deviations.json。',
                  f'累计原始读取上界{spent + original_bytes:,}/{LIMIT:,}字节；大文件读取同时SHA，不为核验再次读轨迹。',
                  '下一步影响：仅补齐既有重复记录的描述性资格；E2仍为历史prefix回放，不调整阈值或启动下一实验。'])
    (OUT / '简明中文结论.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(OUT / 'execution_done.json', {'status': 'PASS' if not errors else 'ERRORS', 'stages': 12,
                                     'cumulative_read_upper_bound_bytes': spent + original_bytes})
    print(json.dumps({'status': 'PASS' if not errors else 'ERRORS', 'errors': errors,
                      'read_upper_bound_MiB': (spent + original_bytes) / (1024 * 1024),
                      'unique_successful_trace_call_IDs': global_unique_calls,
                      'summary': summaries}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--selfcheck']:
        raise SystemExit(0 if selfcheck() else 1)
    if sys.argv[1:]:
        raise SystemExit('Only --selfcheck or no arguments are allowed.')
    main()
