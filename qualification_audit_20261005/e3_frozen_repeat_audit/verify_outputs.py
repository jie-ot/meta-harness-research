"""Verify E3 deliverables only; never reread original project evidence."""
import csv
import hashlib
import json
from pathlib import Path

out = Path(__file__).resolve().parent


def read_csv(name):
    with (out / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


rows = read_csv('repeat_summary.csv')
stages = read_csv('stages.csv')
qual = read_csv('qualification.csv')
budget = json.loads((out / 'read_budget.json').read_text(encoding='utf-8'))
integrity = json.loads((out / 'integrity_checks.json').read_text(encoding='utf-8'))
reuse = json.loads((out / 'h0_deduplication.json').read_text(encoding='utf-8'))
checks = {'exactly_six_phase_pairs': len(rows) == len(qual) == 6,
          'exactly_twelve_unique_stage_records': len(stages) == len({x['stage'] for x in stages}) == 12,
          'read_upper_bound_within_limit': budget['total_original_read_upper_bound_bytes'] <= budget['limit_bytes'],
          'no_recovery_large_trace_reads': budget['full_trace_files_read_in_recovery'] == 0,
          'all_saved_model_groups_explicit_gpt_oss': all(x['solver'] == 'openrouter/openai/gpt-oss-120b' for x in stages),
          'all_sources_off_cache_and_zero_hits': all(x['cache_mode'] == 'off' and x['cache_hits'] == '0' for x in stages),
          'no_experimental_model_calls': integrity['new_experimental_model_calls'] == 0,
          'incomplete_item_qualification_explicit': integrity['full_item_qualification'] == 'INCOMPLETE',
          'small_metadata_checks_passed': integrity['small_metadata_integrity'] == 'PASS',
          'all_repeat_qualifications_incomplete': all(x['category'] == '不匹配或资料不全' for x in qual),
          'copied_stages_never_new_calls': all(x['new_model_calls'] == 0 for x in reuse['copied_stages']),
          'shared_feedback_deduplicated': len({x['dedup_key'] for x in reuse['copied_stages'] if x['phase'] == 'feedback'}) == 1}
stage_map = {x['stage']: x for x in stages}
for r in rows:
    a, b = [stage_map[f"{r['version']}/{rep}/{r['phase']}"] for rep in (1, 2)]
    p1 = 100 * int(a['correct']) / int(a['total'])
    p2 = 100 * int(b['correct']) / int(b['total'])
    delta = p2 - p1
    checks[r['version'] + '/' + r['phase'] + '_saved_score_arithmetic'] = (
        int(r['n_repetitions']) == 2 and int(r['items_per_repetition']) == 100
        and float(r['accuracy_rep1_pp']) == p1 and float(r['accuracy_rep2_pp']) == p2
        and float(r['delta_rep2_minus_rep1_pp']) == delta
        and float(r['range_min_pp']) == min(p1, p2) and float(r['range_max_pp']) == max(p1, p2)
        and all((r[f'abs_delta_ge_{n}pp'] == 'True') == (abs(delta) >= n) for n in (1, 2, 5)))
    checks[r['version'] + '/' + r['phase'] + '_no_invented_item_scores'] = all(
        r[k] == 'missing' for k in ('CC_item_verified', 'CW_item_verified', 'WC_item_verified',
                                    'WW_item_verified', 'normalized_label_flips', 'raw_text_flips'))
    if r['phase'] == 'audit':
        checks[r['version'] + '_old_index_net_change_only'] = (
            int(r['WC_existing_index_only']) - int(r['CW_existing_index_only']) == int(r['net_correct_change']))
artifacts = {}
for p in out.iterdir():
    if p.is_file() and p.name not in ('output_review.json',):
        artifacts[p.name] = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
          'original_files_reread': 0, 'artifacts': artifacts}
(out / 'output_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'checks': len(checks),
                  'failures': [k for k, v in checks.items() if not v],
                  'deliverables': list(artifacts),
                  'read_upper_bound_MiB': budget['total_original_read_upper_bound_bytes'] / (1024 * 1024)}, ensure_ascii=False))
