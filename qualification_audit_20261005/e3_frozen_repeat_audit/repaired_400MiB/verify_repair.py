"""Verify repaired E3 outputs only; zero original evidence reads."""
import csv
import hashlib
import json
from pathlib import Path

out = Path(__file__).resolve().parent


def js(p):
    return json.loads(p.read_text(encoding='utf-8'))


def csv_read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


summary = csv_read(out / 'repeat_summary.csv')
qualification = csv_read(out / 'qualification.csv')
aligned = csv_read(out / 'aligned_items.csv')
stage_rows = csv_read(out / 'stages.csv')
stages = {x['stage']: x for x in stage_rows}
budget = js(out / 'read_budget.json')
integrity = js(out / 'integrity_checks.json')
manifest = js(out / 'input_manifest.json')
reuse = js(out / 'h0_deduplication.json')
checks = {'six_summary_rows': len(summary) == len(qualification) == 6,
          'twelve_unique_stages': len(stage_rows) == len(stages) == 12,
          'six_hundred_aligned_item_pairs': len(aligned) == 600,
          'within_cumulative_400MiB': budget['total_original_read_upper_bound_bytes'] <= 400 * 1024 * 1024,
          'read_budget_matches_manifest': sum(x['read_bytes'] for x in manifest) == budget['new_original_read_bytes'],
          'hash_and_parse_same_pass': budget['hashing_same_stream_as_parse'] is True,
          'zero_experimental_calls': integrity['new_experimental_model_calls'] == 0,
          'historical_model_group_kept': all(x['model'] == 'openrouter/openai/gpt-oss-120b' for x in stage_rows),
          'independence_stays_unknown': integrity['scientific_independence'].startswith('unknown'),
          'no_first_class_assignment': all(x['category'] == '匹配但缓存或独立性未知' for x in qualification),
          'all_raw_API_full_reads_excluded': budget['raw_calls_files_fully_read'] == 0,
          'all_original_sources_stable': all(x['unchanged_size_mtime'] for x in js(out / 'source_stat_recheck.json')),
          'selfcheck_passed': js(out / 'parser_selfcheck.json')['status'] == 'PASS',
          'all_stage_integrity_passed': integrity['status'] == 'PASS' and not integrity['errors'],
          'exact_1200_distinct_trace_calls': reuse['noise_stage_call_ID_sum'] == reuse['noise_stage_call_ID_unique'] == 1200,
          'no_injected_new_calls': all(x['new_model_calls'] == 0 for x in reuse['copied_stages']),
          'shared_H0_feedback_deduplicated': len({x['dedup_key'] for x in reuse['copied_stages'] if x['phase'] == 'feedback'}) == 1}
records = {}
for p in (out / 'stage_checkpoints').glob('*.json'):
    cp = js(p)
    root = out.parents[2]
    saved = root / cp['compact_records']
    # cp paths are relative to meta-harness; resolve from the same project root.
    project = out.parents[2]
    saved = project / cp['compact_records']
    data = saved.read_bytes()
    rows = [json.loads(x) for x in data.splitlines() if x.strip()]
    records[cp['stage']] = rows
    checks[cp['stage'] + '_checkpoint_integrity'] = (
        hashlib.sha256(data).hexdigest() == cp['compact_records_sha256']
        and cp['rows'] == len(rows) == 100 and not cp['parse_errors']
        and all(cp['checks'].values()))
for row in summary:
    v, phase = row['version'], row['phase']
    a, b = [records[f'{v}/{rep}/{phase}'] for rep in (1, 2)]
    bm = {x['item_id']: x for x in b}
    counts = dict(CC=0, CW=0, WC=0, WW=0)
    label_flips = raw_flips = actual_equal = constructed_equal = 0
    for x in a:
        y = bm[x['item_id']]
        transition = ('C' if x['was_correct_saved'] else 'W') + ('C' if y['was_correct_saved'] else 'W')
        counts[transition] += 1
        label_flips += int(x['canonical_prediction_derived'] != y['canonical_prediction_derived'])
        raw_flips += int(x['prediction_sha256'] != y['prediction_sha256'])
        actual_equal += int(x['actual_prompt_sha256'] == y['actual_prompt_sha256'])
        constructed_equal += int(x['constructed_prompt_sha256'] == y['constructed_prompt_sha256'])
    p1 = int(stages[f'{v}/1/{phase}']['correct'])
    p2 = int(stages[f'{v}/2/{phase}']['correct'])
    checks[v + '/' + phase + '_independent_recount'] = (
        all(int(row[k]) == n for k, n in counts.items()) and sum(counts.values()) == 100
        and counts['CC'] + counts['CW'] == p1 and counts['CC'] + counts['WC'] == p2
        and float(row['accuracy_rep1_pp']) == p1 and float(row['accuracy_rep2_pp']) == p2
        and float(row['delta_rep2_minus_rep1_pp']) == p2 - p1 == counts['WC'] - counts['CW']
        and float(row['range_min_pp']) == min(p1, p2) and float(row['range_max_pp']) == max(p1, p2)
        and int(row['canonical_label_set_flips']) == label_flips
        and int(row['raw_prediction_text_flips']) == raw_flips
        and int(row['actual_prompt_equal_items']) == actual_equal == 100
        and int(row['constructed_prompt_equal_items']) == constructed_equal == 100
        and label_flips <= raw_flips)
    checks[v + '/' + phase + '_descriptive_thresholds'] = all(
        (row[f'abs_delta_ge_{n}pp'] == 'True') == (abs(p2 - p1) >= n) for n in (1, 2, 5))
checks['all_1200_derived_labels_consistent_with_saved_decisions'] = all(
    x['canonical_agrees_saved_correct'] is True for rows in records.values() for x in rows)
artifacts = {}
for p in out.rglob('*'):
    if p.is_file() and p.name != 'output_review.json':
        artifacts[p.relative_to(out).as_posix()] = {'bytes': p.stat().st_size,
                                                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
          'original_evidence_reads': 0, 'artifacts': artifacts}
(out / 'output_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'checks': len(checks),
                  'failures': [k for k, ok in checks.items() if not ok],
                  'artifacts': len(artifacts), 'original_evidence_reads': 0}, ensure_ascii=False))
