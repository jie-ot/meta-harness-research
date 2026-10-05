"""Bounded, offline audit parser. Standard library only; no project imports."""
from pathlib import Path
import ast, csv, hashlib, json, re

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'reference_examples/text_classification/logs/20260925_200856'
PILOT = ROOT / 'lawbench_rsi_v3'
seen = {}

def read(p):
    b = p.read_bytes()
    seen[str(p.relative_to(ROOT))] = hashlib.sha256(b).hexdigest()
    return b.decode('utf-8-sig')

def obj(p):
    return json.loads(read(p))

def save(name, x):
    (OUT / name).write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')

def table(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

history = [json.loads(s) for s in read(OLD / 'evolution_summary.jsonl').splitlines() if s.strip()]
snapshots, parents, sessions = {}, {}, []
for d in sorted((OLD / 'claude_sessions').iterdir()):
    m = re.search(r'_iter(\d+)$', d.name)
    if not m: continue
    it = int(m[1]); meta = obj(d / 'meta.json')
    cmd = meta.get('command', [])
    effort = 'missing'
    if isinstance(cmd, list) and '--effort' in cmd: effort = cmd[cmd.index('--effort') + 1]
    sessions.append({'run_id': OLD.name, 'round': it, 'model': meta.get('model'), 'effort': effort,
                     'session_id': meta.get('session_id'), 'timestamp': meta.get('timestamp'),
                     'proposer_cost_usd': meta.get('cost_usd'), 'exit_code': meta.get('exit_code'),
                     'source': str((d / 'meta.json').relative_to(ROOT))})
    for p in sorted((d / 'tools').glob('*_Read.txt')):
        with p.open(encoding='utf-8') as f: header = f.readline()
        if 'frontier_val.json' not in header or '[ERROR]' in header: continue
        content = read(p).split('--- output ---', 1)[-1]
        content = re.sub(r'(?m)^\s*\d+\t', '', content)
        try:
            x = json.JSONDecoder().raw_decode(content[content.index('{'):])[0]
            snapshots.setdefault(it, (x, str(p.relative_to(ROOT))))
        except (ValueError, IndexError): pass
    for p in sorted((d / 'tools').glob('*_Write.txt')):
        with p.open(encoding='utf-8') as f: header = f.readline()
        if 'pending_eval.json' not in header: continue
        content = read(p).split('content:', 1)[-1].lstrip()
        try:
            x = json.JSONDecoder().raw_decode(content)[0]
            for c in x.get('candidates', []): parents[(it, c['name'])] = (c, str(p.relative_to(ROOT)))
        except (ValueError, KeyError): pass

final_frontier = obj(OLD / 'frontier_val.json')
runtime_log = ROOT / 'reference_examples/text_classification/evolve_out.log'
runtime_scores = {}
runtime_iteration = None
for ln, line in enumerate(read(runtime_log).splitlines(), 1):
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    m = re.search(r'Iteration (\d+).*frontier=(\S+) @ ([0-9.]+)%', line)
    if m:
        runtime_iteration = int(m[1])
        snapshots.setdefault(int(m[1]), ({'_pareto':[{'system':m[2], 'val_accuracy':float(m[3])}]}, str(runtime_log.relative_to(ROOT)) + ':' + str(ln)))
    score = re.search(r'^\s*(\w+): avg_val=([0-9.]+)%', line)
    if score and runtime_iteration is not None:
        runtime_scores[(runtime_iteration, score[1])] = (float(score[2]), str(runtime_log.relative_to(ROOT)) + ':' + str(ln))
candidates, rounds, evaluations = [], [], []
best = snapshots.get(1, ({}, 'missing'))[0].get('_pareto', [{}])[0].get('val_accuracy')
for it in sorted({x['iteration'] for x in history}):
    rows = [x for x in history if x['iteration'] == it]
    before, source = snapshots.get(it, ({}, 'missing'))
    after, after_source = snapshots.get(it + 1, ({}, 'missing')) if it < 20 else (final_frontier, str((OLD / 'frontier_val.json').relative_to(ROOT)))
    bi = before.get('_pareto', [{}])[0].get('system', 'missing')
    ai = after.get('_pareto', [{}])[0].get('system', 'missing') if it < 20 else 'missing_contemporaneous_final_snapshot'
    def original_value(x):
        if x.get('recovered_after_search'): return runtime_scores.get((it, x['system']), ('missing_import_failed', 'missing'))[0]
        return runtime_scores.get((it, x['system']), (x['avg_val'], 'evolution_summary'))[0]
    if best is not None: best = max([best] + [original_value(x) for x in rows if isinstance(original_value(x), (int, float))])
    rd = {'group': 'text_classification_reproduction', 'run_id': OLD.name, 'round': it,
          'new_candidates': len(rows), 'incumbent_before': bi, 'incumbent_after': ai,
          'raw_candidates': '; '.join(f"{x['system']}={original_value(x)}" for x in rows),
          'best_so_far_recorded_validation_pp': best, 'selection': 'macro validation / Pareto, then context length',
          'before_evidence': source, 'after_evidence': after_source}
    if it == 20: rd['current_final_frontier_incumbent'] = final_frontier['_pareto'][0]['system']
    for ds in ['USPTO', 'Symptom2Disease', 'LawBench']:
        rd[f'{ds}_incumbent_after'] = after.get(ds, {}).get('best_system', 'missing')
        tp = OLD / 'results' / ds / ai / 'gpt-oss-120b/test.json'
        rd[f'fixed_global_incumbent_{ds}_test_accuracy'] = obj(tp).get('accuracy') if tp.exists() else 'missing'
    rounds.append(rd)
    for x in rows:
        c, ps = parents.get((it, x['system']), ({}, 'missing'))
        candidates.append({'group': 'text_classification_reproduction', 'run_id': OLD.name,
                           'round': it, 'candidate': x['system'], 'axis': x.get('axis'),
                           'declared_parent': c.get('base_system', 'missing'), 'parent_evidence': ps,
                           'parent_equals_observed_incumbent': c.get('base_system') == bi if c else 'missing',
                           'actual_copy_provenance': 'not verified; declared parent only',
                           'raw_validation_pp': original_value(x), 'current_summary_validation_pp': x['avg_val'],
                           'recovered_after_search': x.get('recovered_after_search', False),
                           'raw_validation_evidence': runtime_scores.get((it,x['system']), (None, 'original summary if unrecovered'))[1],
                           'evaluation_status': 'benchmark_failed_logged_zero' if x['system']=='hard_buffer_memory' else ('import_failed_no_score' if x['system']=='per_label_recent_memory' else 'recorded_evaluation'),
                           'current_summary_delta_pp': x.get('delta'),
                           'best_so_far_recorded_validation_pp': best,
                           'solver': 'openrouter/openai/gpt-oss-120b', 'proposer': 'claude-opus-5-5-code',
                           'source': str((OLD / 'evolution_summary.jsonl').relative_to(ROOT)) + ':' + str(history.index(x) + 1)})

for run in ['D0_a', 'D0_b', 'D100_a', 'D100_b', 'D100_injected_a', 'D100_injected_b']:
    control = PILOT / 'external/control' / run
    package = PILOT / 'runs' / run / 'text_classification'
    prev = 'confusion_disambiguation_memory'
    for it in [1, 2]:
        cp_path = control / f'round{it}_checkpoint.json'
        if not cp_path.exists(): continue
        cp = obj(cp_path); pending = obj(control / f'proposer_round{it}/pending_eval.json')
        pm_path = control / f'proposer_round{it}/result.json'; pm = obj(pm_path)
        sessions.append({'run_id': run, 'round': it, **{k: pm.get(k) for k in ['model','effort','session_id','api_requests','logical_calls','failed_api_requests','requested_models','cost','success']}, 'source': str(pm_path.relative_to(ROOT))})
        rd = {'group': 'lawbench_injected' if 'injected' in run else 'lawbench_pilot', 'run_id': run,
              'round': it, 'D': cp['D'], 'new_candidates': len(pending['candidates']), 'N': cp['N'],
              'incumbent_before': prev, 'incumbent_after': cp['selected'], 'selection': 'score correct, ties earliest candidate',
              'raw_candidates': '; '.join(f"{x['candidate']}={x.get('correct')}" for x in cp['candidates']),
              'best_so_far_recorded_validation_pp': max(x['correct'] for x in cp['candidates'] if x.get('correct') is not None),
              'checkpoint_time_utc': cp['created_at_utc'], 'checkpoint_evidence': str(cp_path.relative_to(ROOT))}
        for c in pending['candidates']:
            name = c['name']; score_path = package / 'history' / name / 'score/result.json'
            audit_path = PILOT / 'external/audit' / run / name / 'result.json'
            sc, au = obj(score_path), obj(audit_path)
            candidates.append({'group': rd['group'], 'run_id': run, 'round': it, 'D': cp['D'],
                               'slot': c.get('slot'), 'candidate': name, 'axis': c.get('axis'),
                               'declared_parent': c.get('base_system', 'missing'), 'parent_evidence': str((control / f'proposer_round{it}/pending_eval.json').relative_to(ROOT)),
                               'parent_equals_observed_incumbent': c.get('base_system') == prev,
                               'actual_copy_provenance': 'not verified; declared parent only',
                               'raw_validation_pp': 100 * sc['accuracy'], 'score_correct': sc['correct'],
                               'score_total': sc['total'], 'audit_pp': 100 * au['accuracy'],
                               'selected_by_validation': cp['selected'] == name, 'code_sha256': c.get('code_hash'),
                               'solver': sc['model'], 'proposer': pm['model'], 'effort': pm['effort']})
        rd['raw_new_candidates'] = '; '.join(f"{c['name']}={next(x['correct'] for x in cp['candidates'] if x['candidate']==c['name'])}" for c in pending['candidates'])
        sel = cp['selected']; ap = PILOT / 'external/audit' / run / sel / 'result.json'
        rd['fixed_incumbent_audit_pp'] = 100 * obj(ap)['accuracy'] if ap.exists() else 'missing; shared H0 frozen repeats, see noise_summary'
        rounds.append(rd); prev = sel
    for d in sorted((package / 'history').iterdir()):
        for phase in ['train', 'score', 'feedback', 'audit']:
            p = (PILOT / 'external/audit' / run / d.name if phase == 'audit' else d / phase) / 'result.json'
            if not p.exists(): continue
            x = obj(p); u = x.get('usage', {})
            evaluations.append({'run_id': run, 'candidate': d.name, 'phase': phase,
                                **{k: x.get(k) for k in ['total','correct','accuracy','timestamp','model','seed','cache_mode','code_hash','memory_hash','manifest_hash','persistent_prediction_state_changed']},
                                **{k: u.get(k) for k in ['logical_calls','api_requests','cache_hits','failed_api_requests','retry_attempts','unreported_failed_cost_count','cost_usd']},
                                'model_config': json.dumps(x.get('model_config'), ensure_ascii=False), 'result_source': str(p.relative_to(ROOT))})

for ds in ['USPTO', 'Symptom2Disease', 'LawBench']:
    for p in sorted((OLD / ds).glob('*/gpt-oss-120b/val.json')) + sorted((OLD / 'results' / ds).glob('*/gpt-oss-120b/test.json')):
        x = obj(p)
        evaluations.append({'run_id': OLD.name, 'candidate': p.parent.parent.name, 'phase': 'val' if p.name == 'val.json' else 'test',
                            'dataset':ds, **{k:x.get(k) for k in ['total','correct','accuracy','timestamp','model','seed','llm_calls','estimated_cost_usd']},
                            'result_source':str(p.relative_to(ROOT)), 'history_note':'current archived result; do not overwrite original evolution_summary scores with recovery scores'})
recovery = []
rec_dir = OLD.parent / '20260925_200856_recovery_20260927_184423'
for ds in ['USPTO', 'Symptom2Disease', 'LawBench']:
    for p in sorted((rec_dir / ds).glob('*/gpt-oss-120b/val.json')):
        x = obj(p)
        recovery.append({'run_id':rec_dir.name, 'candidate':p.parent.parent.name, 'dataset':ds,
                         **{k:x.get(k) for k in ['accuracy','correct','total','timestamp','model','seed']},
                         'source':str(p.relative_to(ROOT)), 'used_by_later_original_rounds':False})

manifest_path = PILOT / 'external/prepared_data_v3/manifest_v3.json'
manifest = obj(manifest_path); splits = {}; idsets = {}
for name, spec in manifest['splits'].items():
    p = manifest_path.parent / spec['file']; text = read(p)
    rows = [json.loads(s) for s in text.splitlines() if s.strip()]
    ids = [x['item_id'] for x in rows]; idsets[name] = set(ids)
    splits[name] = {'count':len(rows), 'sha256':seen[str(p.relative_to(ROOT))], 'declared_sha256':spec['sha256'],
                    'hash_matches':seen[str(p.relative_to(ROOT))] == spec['sha256'], 'ids_in_execution_order':ids,
                    'legacy_counts':{str(v):sum(x.get('legacy') == v for x in rows) for v in {x.get('legacy') for x in rows}}}
save('split_evidence.json', {'manifest_sha256':seen[str(manifest_path.relative_to(ROOT))], 'metadata':{k:v for k,v in manifest.items() if k != 'splits'},
                             'splits':splits, 'item_id_overlap':{a+' / '+b:len(idsets[a]&idsets[b]) for a in idsets for b in idsets if a < b}})
table('candidates.csv', candidates); table('rounds.csv', rounds); table('evaluations.csv', evaluations)
table('post_search_recovery.csv', recovery)
save('proposer_sessions.json', sessions); save('source_hashes.json', seen)
print(json.dumps({'output':str(OUT), 'candidate_rows':len(candidates), 'round_rows':len(rounds),
                  'evaluation_rows':len(evaluations), 'old_frontier_snapshots':len(snapshots),
                  'old_candidate_parent_records':len(parents), 'proposer_records':len(sessions),
                  'post_search_recovery_rows':len(recovery)}, ensure_ascii=False))
