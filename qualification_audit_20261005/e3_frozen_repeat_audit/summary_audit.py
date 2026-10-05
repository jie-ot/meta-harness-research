"""Bounded E3 recovery: saved summaries only; no project imports or large reads.

Uses only the Python standard library. Retains incomplete item qualification.
The prior large-file pass exhausted the allowed practical read budget, so this
script refuses to read traces, calls, caches or initial-memory contents.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT / 'lawbench_rsi_v3'
LIMIT = 160 * 1024 * 1024
ledger = json.loads((OUT / 'aborted_full_pass_read_ledger.json').read_text(encoding='utf-8'))
SPENT = ledger['total_original_read_upper_bound_before_recovery'] + ledger['extra_small_recovery_read_reserve_bytes']
if (OUT / 'integrity_checks.json').exists():
    raise SystemExit('This bounded audit has already completed. Do not reread original files.')
VERSIONS = {'R4B': 'confusion_disambiguation_memory',
            'R7A': 'adaptive_tokenizer_confusion_memory',
            'R12B': 'discriminative_similarity_memory'}
ENGINE = ['llm.py', 'inner_loop.py', 'memory_system.py', 'pilot_observation.py',
          'data/api.py', 'data/loaders.py', 'data/evaluators.py']
MODEL_KEYS = ['solver', 'temperature', 'max_tokens', 'system_prompt', 'training_mode',
              'training_batch_size', 'training_seed', 'evaluation_workers', 'max_api_retries']
CATEGORY = '不匹配或资料不全'
allow = {}


def rel(p):
    return p.relative_to(ROOT).as_posix()


def permit(p, purpose, required=True):
    p = p.resolve()
    p.relative_to(ROOT)
    if not p.is_file():
        if required:
            raise FileNotFoundError(rel(p))
        return
    s = p.stat()
    allow[p] = {'path': rel(p), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'purpose': purpose}


for v in VERSIONS:
    for rep in (1, 2):
        for phase in ('score', 'audit'):
            d = BASE / 'external/noise' / v / str(rep) / phase
            for n in ('result.json', 'complete.json', 'start.json', 'attempt_times.jsonl', 'failure.json'):
                permit(d / n, '12-stage small metadata', required=n in ('result.json', 'complete.json', 'start.json'))
for candidate in VERSIONS.values():
    permit(BASE / 'engine/text_classification/agents' / (candidate + '.py'), 'agent code and fixed template')
for n in ENGINE:
    permit(BASE / 'engine/text_classification' / n, 'static framework and model-call parameters')
for n in ('worker.py', 'run_pilot.py', 'state.py', 'injected.py'):
    permit(BASE / 'pilot' / n, 'static reset/resume/reuse')
for n in ('noise_summary.csv', 'frozen_prompt_consistency.json'):
    permit(BASE / 'analysis' / n, 'existing summary evidence')
manifest_path = BASE / 'external/prepared_data_v3/manifest_v3.json'
permit(manifest_path, 'locked split manifest')
for phase in ('score', 'audit'):
    permit(manifest_path.parent / (phase + '.jsonl'), 'split byte/order hashes; no grading')
for run in ('D0_a', 'D0_b', 'D100_a'):
    permit(BASE / 'external/control' / run / 'baseline_cache_recovery.json', 'cache provenance metadata only')
for run in ('D100_injected_a', 'D100_injected_b'):
    permit(BASE / 'external/control' / run / 'reused_h0.json', 'copy provenance metadata only')
if SPENT + sum(x['bytes'] for x in allow.values()) > LIMIT:
    raise SystemExit('Recovery allowlist exceeds cumulative read budget.')

raw = {}
hashes = {}
inputs = []
for p, record in allow.items():
    before = p.stat()
    if (before.st_size, before.st_mtime_ns) != (record['bytes'], record['mtime_ns']):
        raise ValueError('Input changed before read: ' + rel(p))
    data = p.read_bytes()
    after = p.stat()
    if len(data) != before.st_size or (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise ValueError('Input changed during read: ' + rel(p))
    raw[p] = data
    hashes[p] = hashlib.sha256(data).hexdigest()
    inputs.append({**record, 'sha256': hashes[p], 'unchanged_during_read': True})


def js(p):
    return json.loads(raw[p.resolve()])


def dump(n, value):
    (OUT / n).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def table(n, rows):
    with (OUT / n).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


old = list(csv.DictReader(raw[(BASE / 'analysis/noise_summary.csv').resolve()].decode('utf-8-sig').splitlines()))
oldmap = {(x['version'], int(x['repetition'])): x for x in old}
m = js(manifest_path)
split_evidence = {}
for phase in ('score', 'audit'):
    p = manifest_path.parent / (phase + '.jsonl')
    rows = [json.loads(x) for x in raw[p.resolve()].splitlines() if x.strip()]
    ids = [x['item_id'] for x in rows]
    split_evidence[phase] = {'count': len(ids), 'data_sha256': hashes[p.resolve()],
                            'hash_matches_manifest': hashes[p.resolve()] == m['splits'][phase]['sha256'],
                            'order_matches_manifest': ids == m['splits'][phase]['item_ids'],
                            'item_order_hash': hashlib.sha256(json.dumps(ids, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest(),
                            'runtime_input_alignment': 'unknown: not retained from aborted detailed pass'}

static = {'agents': {}, 'framework': {}, 'pilot_functions': {},
          'reset_and_resume': 'worker loads fresh agent, set_state(initial memory); no evaluation learning; resumes completed per-item records after signature validation.',
          'predict_state_limit': 'Saved get_state comparison false does not observe all transient state or expanded prompts.',
          'API_kwargs': 'Solver gpt-oss-120b, temperature 0, max_tokens 16384, system Reasoning: medium. training_seed 42 is not a provider/API seed. Completion timeout 600 seconds, SDK num_retries 0, wrapper permits up to 3 retry attempts.',
          'raw_API_logs': 'not read; summary failure/retry accounting not reconciled against individual API events',
          'initial_memory_content_hash_verification': 'not retained from aborted detailed pass; matching declared digest is checked, contents not reread'}
for v, candidate in VERSIONS.items():
    p = BASE / 'engine/text_classification/agents' / (candidate + '.py')
    tree = ast.parse(raw[p.resolve()].decode('utf-8-sig'))
    template = {}
    functions = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            for t in n.targets:
                if isinstance(t, ast.Name) and 'PROMPT' in t.id:
                    template[t.id] = {'line': n.lineno, 'sha256_utf8': hashlib.sha256(n.value.value.encode('utf-8')).hexdigest()}
        if isinstance(n, ast.FunctionDef) and n.name in ('predict', 'get_state', 'set_state', '_build_parts'):
            functions[n.name] = {'start': n.lineno, 'end': n.end_lineno}
    static['agents'][v] = {'candidate': candidate, 'code_path': rel(p), 'code_sha256': hashes[p.resolve()],
                            'fixed_templates': template, 'function_lines': functions}
for name in ENGINE:
    p = BASE / 'engine/text_classification' / name
    static['framework'][name] = {'path': rel(p), 'sha256': hashes[p.resolve()]}
for name in ('worker.py', 'run_pilot.py', 'state.py', 'injected.py'):
    p = BASE / 'pilot' / name
    tree = ast.parse(raw[p.resolve()].decode('utf-8-sig'))
    static['pilot_functions'][rel(p)] = {n.name: {'start': n.lineno, 'end': n.end_lineno}
                                         for n in tree.body if isinstance(n, ast.FunctionDef)}

reuse = {'recovered_caches': [], 'copied_stages': [],
         'policy': 'Only 12 noise stages counted; restored H0 caches and injected stage copies never increase n.',
         'copy_hash_verification': 'declared source_complete/result hashes retained, copied stage payloads not read'}
for run in ('D0_a', 'D0_b', 'D100_a'):
    p = BASE / 'external/control' / run / 'baseline_cache_recovery.json'
    x = js(p)
    counts = {}
    for f in x['files']:
        counts[f['phase']] = counts.get(f['phase'], 0) + 1
    reuse['recovered_caches'].append({'run': run, 'path': rel(p), 'source': x['source'],
                                     'entries_by_phase': counts, 'source_execution_ID': 'unknown',
                                     'additional_repetitions': 0})
for run in ('D100_injected_a', 'D100_injected_b'):
    p = BASE / 'external/control' / run / 'reused_h0.json'
    x = js(p)
    for phase, entry in x['stages'].items():
        reuse['copied_stages'].append({'destination': run, 'phase': phase, **entry,
                                       'dedup_key': entry['source_complete_hash']})

stages = {}
stage_rows = []
checks = []
errors = []
compare_fields = ['candidate', 'D', 'phase', 'manifest_hash', 'item_ids_hash', 'code_hash',
                  'memory_hash', 'model_config', 'cache_mode', 'cache_dir', 'engine_hashes']
for v, candidate in VERSIONS.items():
    for rep in (1, 2):
        for phase in ('score', 'audit'):
            d = BASE / 'external/noise' / v / str(rep) / phase
            result, complete, start = [js(d / f) for f in ('result.json', 'complete.json', 'start.json')]
            sig = complete['signature']
            usage = result['usage']
            check = {'stage': f'{v}/{rep}/{phase}',
                     'result_hash_matches_complete': hashes[(d / 'result.json').resolve()] == complete['files']['result.json'],
                     'start_signature_matches_complete': start['signature'] == sig,
                     'result_usage_matches_complete': usage == complete['usage'],
                     'code_matches_declared': static['agents'][v]['code_sha256'] == sig['code_hash'],
                     'engine_matches_declared': all(static['framework'][n]['sha256'] == h for n, h in sig['engine_hashes'].items()),
                     'split_order_matches_declared': split_evidence[phase]['item_order_hash'] == sig['item_ids_hash'],
                     'manifest_matches_declared': hashes[manifest_path.resolve()] == sig['manifest_hash'],
                     'summary_matches_existing_index': abs(100 * result['accuracy'] - float(oldmap[(v, rep)][phase + '_pp'])) < 1e-9,
                     'saved_correct_total_matches_accuracy': abs(result['correct'] / result['total'] - result['accuracy']) < 1e-9}
            checks.append(check)
            if any(value is False for value in check.values()):
                errors.append(check)
            ap = d / 'attempt_times.jsonl'
            attempts = [json.loads(x) for x in raw.get(ap.resolve(), b'').splitlines() if x.strip()]
            fp = d / 'failure.json'
            failure = js(fp) if fp.resolve() in raw else None
            key = f'{v}/{rep}/{phase}'
            stages[key] = {'result': result, 'signature': sig}
            stage_rows.append({'stage': key, 'run_id': result['run_id'], 'candidate': candidate,
                               'phase': phase, 'rep': rep, 'correct': result['correct'], 'total': result['total'],
                               'accuracy_pp': 100 * result['accuracy'], 'start_utc': start['started_at_utc'],
                               'result_utc': result['timestamp'], 'complete_utc': complete['completed_at_utc'],
                               'solver': result['model'], 'model_config_json': json.dumps(sig['model_config'], ensure_ascii=False, sort_keys=True),
                               'code_hash': sig['code_hash'], 'initial_memory_hash_declared': sig['memory_hash'],
                               'engine_hashes_json': json.dumps(sig['engine_hashes'], sort_keys=True),
                               'item_order_hash': sig['item_ids_hash'], 'manifest_hash': sig['manifest_hash'],
                               'cache_mode': sig['cache_mode'], 'cache_dir': sig['cache_dir'], 'cache_hits': usage['cache_hits'],
                               'logical_calls': usage['logical_calls'], 'api_requests': usage['api_requests'],
                               'failed_API_requests': usage['failed_api_requests'], 'retry_attempts': usage['retry_attempts'],
                               'incomplete_attempts_json': json.dumps(usage['incomplete_attempts']),
                               'nonbillable_credit_rejections': usage['nonbillable_credit_rejections'],
                               'unreported_failed_cost_count': usage['unreported_failed_cost_count'],
                               'reported_cost_usd': usage['reported_usd'], 'estimated_cost_usd': usage['estimated_usd'],
                               'attempt_time_rows': len(attempts), 'retained_failure_HTTP_status': failure.get('http_status') if failure else None,
                               'persistent_prediction_state_changed': result['persistent_prediction_state_changed'],
                               'actual_expanded_prompt_match': 'unknown: detailed results not retained',
                               'provider_response_ID_and_backend': 'missing', 'API_seed': 'missing',
                               'source': rel(d / 'result.json')})

summary = []
qualification = []
for v in VERSIONS:
    for phase in ('score', 'audit'):
        a, b = [stages[f'{v}/{rep}/{phase}'] for rep in (1, 2)]
        signatures_match = all(a['signature'][k] == b['signature'][k] for k in compare_fields)
        r1, r2 = a['result'], b['result']
        p1, p2 = [100 * r['correct'] / r['total'] for r in (r1, r2)]
        delta = p2 - p1
        summary.append({'version': v, 'phase': phase, 'n_repetitions': 2, 'items_per_repetition': 100,
                        'accuracy_rep1_pp': p1, 'accuracy_rep2_pp': p2, 'delta_rep2_minus_rep1_pp': delta,
                        'range_min_pp': min(p1, p2), 'range_max_pp': max(p1, p2),
                        'net_correct_change': r2['correct'] - r1['correct'],
                        'CC_item_verified': 'missing', 'CW_item_verified': 'missing',
                        'WC_item_verified': 'missing', 'WW_item_verified': 'missing',
                        'WC_existing_index_only': int(oldmap[(v, 1)]['wrong_to_correct']) if phase == 'audit' else 'missing',
                        'CW_existing_index_only': int(oldmap[(v, 1)]['correct_to_wrong']) if phase == 'audit' else 'missing',
                        'normalized_label_flips': 'missing', 'raw_text_flips': 'missing',
                        'abs_delta_ge_1pp': abs(delta) >= 1, 'abs_delta_ge_2pp': abs(delta) >= 2,
                        'abs_delta_ge_5pp': abs(delta) >= 5,
                        'evidence_level': 'verified saved aggregate; item qualification incomplete',
                        'qualification': CATEGORY})
        qualification.append({'version': v, 'phase': phase, 'category': CATEGORY,
                              'recorded_protocol_signatures_equal': signatures_match,
                              'same_declared_initial_memory_hash': a['signature']['memory_hash'] == b['signature']['memory_hash'],
                              'cache_mode_rep1': a['signature']['cache_mode'], 'cache_mode_rep2': b['signature']['cache_mode'],
                              'cache_hits_rep1': r1['usage']['cache_hits'], 'cache_hits_rep2': r2['usage']['cache_hits'],
                              'distinct_run_IDs': r1['run_id'] != r2['run_id'],
                              'persistent_state_changed_rep1': r1['persistent_prediction_state_changed'],
                              'persistent_state_changed_rep2': r2['persistent_prediction_state_changed'],
                              'per_item_ID_and_input_alignment': 'unknown: not persisted from aborted pass',
                              'actual_prompt_equality': 'unknown: not persisted from aborted pass',
                              'trace_call_ID_independence': 'unknown: not persisted from aborted pass',
                              'upstream_cache_or_backend_independence': 'unknown',
                              'notes': '记录配置匹配；实际输入/展开prompt与分别执行资格未完成。此类别表示审计证据不足，不表示已发现项目配置不匹配。'})

source_stats = []
for p, old in allow.items():
    now = p.stat()
    ok = (now.st_size, now.st_mtime_ns) == (old['bytes'], old['mtime_ns'])
    source_stats.append({'path': rel(p), 'unchanged_size_mtime': ok})
    if not ok:
        errors.append({'path': rel(p), 'changed': True})

table('repeat_summary.csv', summary)
table('qualification.csv', qualification)
table('stages.csv', stage_rows)
dump('static_evidence.json', static)
dump('h0_deduplication.json', reuse)
dump('split_evidence.json', split_evidence)
dump('input_manifest.json', inputs)
dump('source_stat_recheck.json', source_stats)
new_bytes = sum(x['bytes'] for x in allow.values())
dump('read_budget.json', {'limit_bytes': LIMIT, 'previous_original_read_upper_bound_bytes': SPENT,
                         'recovery_original_read_bytes': new_bytes,
                         'total_original_read_upper_bound_bytes': SPENT + new_bytes,
                         'full_trace_files_read_in_aborted_pass': 8,
                         'full_trace_files_read_in_recovery': 0,
                         'original_stage_limit': 12, 'raw_API_call_files_read': 0, 'cache_files_read': 0,
                         'large_source_rechecks': 0, 'stat_rechecks_only': True,
                         'conservative_prior_reserves': '2MiB before first parser + first aborted small-input reads + 128KiB + full aborted pass + 128KiB recovery-inspection reserve'})
dump('integrity_checks.json', {'small_metadata_integrity': 'PASS' if not errors else 'FAIL',
                              'full_item_qualification': 'INCOMPLETE', 'errors': errors, 'stage_checks': checks,
                              'stage_count': 12, 'summary_row_count': 6, 'new_experimental_model_calls': 0,
                              'sources_read_only': True, 'no_project_modules_imported_or_executed': True,
                              'no_network_or_subprocess': True})
dump('protocol.json', {'purpose': 'E3 frozen-harness repeat consistency; not paid optimizer-generation variance.',
                       'design': 'three frozen versions x score/audit x two existing repetitions; at most 12 stages',
                       'historical_solver': 'openrouter/openai/gpt-oss-120b',
                       'requested_current_model': 'GPT-6.1 Sol / Max; actual runtime metadata cannot be independently confirmed',
                       'quota': 'No new lookup, reset, purchase, or nested model calls; uses only this chat and offline files.',
                       'read_failure': 'First parser failed on prepared question/input schema before large reads; second failed on repetition/rep CSV column after eight full trace reads. Both counted; no large reread permitted.',
                       'result_level': 'verified 12-stage aggregate metadata, full item qualification unknown',
                       'audit': 'historically exposed old test, not a fresh generalization estimate',
                       'ranking_flips': 'omitted; no verified original cross-harness matched repetition cohort',
                       'no_inference': ['no variance or CI', 'no bootstrap or significance', 'no calibration or predictor', 'no optimized thresholds', 'no new grading'],
                       'threshold_flags': '1/2/5pp descriptive only, equal to 1/2/5 net cases at n=100; opposing flips may cancel',
                       'stop': 'No next experiment or model request.'})

text = [
    'E3：冻结复测的有界离线资格审计（汇总级完成，逐题资格未完成）',
    '1. 目的：描述既有冻结harness两次评价是否一致；不能替代优化器/候选生成方差实验。',
    '2. 设计：R4B/R7A/R12B×score/audit×2重复，严格12原stage；原文件只读，新增实验模型调用0。历史模型均为openrouter/openai/gpt-oss-120b，temperature=0，max_tokens=16384，system_prompt=Reasoning: medium；不能称为SolMax结果。当前模型/effort无可独立确认元数据。',
    '3. 结果（pp为百分点，每次100题，n=2）：',
]
for row in summary:
    text.append(f"{row['version']} {row['phase']}: {row['accuracy_rep1_pp']:g}→{row['accuracy_rep2_pp']:g}，Δ={row['delta_rep2_minus_rep1_pp']:+g}pp，范围[{row['range_min_pp']:g},{row['range_max_pp']:g}]。")
text.extend([
    '六组recorded signature匹配：代码、框架、初始memory声明hash、manifest/题目顺序、模型参数、cache off/null；12个result哈希与complete、start签名、usage及原summary一致。实际执行payload/逐题输入和展开prompt未完成核验。',
    '全部cache_hits=0、persistent_prediction_state_changed=false；不能因此证明服务端无缓存、统计IID或瞬时状态不变。R4B score有历史402后恢复：18/17次非计费拒绝，logical/API=118/117，成功评价仍各100题；其余10个stage各100请求。summary retry=0，incomplete=[]；原API调用日志未核对。',
    '旧noise_summary声明audit的WC/CW依次3/6、5/4、3/2；仅与净准确率差值算术相容，未独立逐题复核，不能当作新核验。规范化标签翻转、CC/CW/WC/WW逐题表和实际prompt一致数均missing/unknown。',
    'H0去重：baseline_cache_recovery来源为own completed H0 measurements，D0_a/b各100条score缓存，D100_a为200条；恢复缓存不增样本。injected两组复用D0_a/b各自score并共用D100_a feedback，new_model_calls=0，不另算重复；副本payload未重读。',
    '4. 结论：六组均归“不匹配或资料不全”，具体是本次逐题资格证据不全，未发现冻结声明配置不匹配。仅支持历史汇总差异描述，不能称固定prompt纯输出噪声或统计独立复测。audit为已暴露old test；不按audit重选，不报告排名翻转、方差、区间、显著性或改善概率。',
    '阻塞：独立解析脚本两次字段错误；第二次发生在8份轨迹读取后，结果未保存。重读会超本步累计预算，故按用户允许的汇总/未知协议结束，不追查独立性。',
    f'累计原始读取上界{SPENT + new_bytes:,}/{LIMIT:,}字节；已计入两次中止及前后小型读取预留。只做stat复核，不重读大文件。',
    '5. 下一步影响：E2 prefix回放仍是历史离线描述；本审计不调整delta/patience，不证实候选生成稳定性，不提供SolMax成绩。最小新增实验调用=0（本步已结束）；任何后续付费实验需另获明确授权，不因本结果自行启动。',
])
(OUT / '简明中文结论.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
print(json.dumps({'small_metadata_integrity': 'PASS' if not errors else 'FAIL',
                  'full_item_qualification': 'INCOMPLETE', 'errors': errors,
                  'total_read_upper_bound': SPENT + new_bytes,
                  'summary': [{k: r[k] for k in ('version', 'phase', 'accuracy_rep1_pp', 'accuracy_rep2_pp', 'delta_rep2_minus_rep1_pp')} for r in summary]},
                 ensure_ascii=False, indent=2))
