"""E2 frozen exploratory replay. Standard library; no project imports or network."""
from pathlib import Path
from fractions import Fraction
import csv, hashlib, io, json

OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent
ROOT = AUDIT.parent
OLD_RUN = '20260925_200856'
DATASETS = ('USPTO', 'Symptom2Disease', 'LawBench')
HORIZON = 19
BASE = 'confusion_disambiguation_memory'
INPUTS = {}

def read(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Input outside authorized workspace')
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8-sig')

def js(path):
    return json.loads(read(path))

def rows(path):
    return list(csv.DictReader(io.StringIO(read(path))))

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(name, data):
    fields = list(dict.fromkeys(k for row in data for k in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)

def patience(values, delta, exact_check=False):
    """One opportunity per whole round. Round 1 initializes; equal never resets."""
    anchor, count = values[0], 0
    trace = [{'round': 1, 'V': str(anchor) if exact_check else anchor,
              'anchor_before': None, 'anchor_after': str(anchor) if exact_check else anchor,
              'reset': False, 'count': 0, 'stop': False, 'decision': 'initialize'}]
    for t, value in enumerate(values[1:], 2):
        previous = anchor
        reset = value > anchor if delta == 0 else value - anchor >= delta
        if reset:
            anchor, count = value, 0
        else:
            count += 1
        stop = count == 3
        trace.append({'round': t, 'V': str(value) if exact_check else value,
                      'anchor_before': str(previous) if exact_check else previous,
                      'anchor_after': str(anchor) if exact_check else anchor,
                      'reset': reset, 'count': count, 'stop': stop,
                      'decision': 'reset' if reset else 'no_reset'})
        if stop:
            return t, True, trace
    return len(values), False, trace

def rule_checks():
    checks = []
    cases = [
        ('equal_rounds_earliest_stop_is_4', [1, 1, 1, 1], 0, 4, True),
        ('strict_delta0_increase_resets', [1, 2, 2, 2, 2], 0, 5, True),
        ('delta2_exact_boundary_resets', [0, 2, 2, 2, 2], 2, 5, True),
        ('delta2_accumulates_from_anchor', [0, 1, 2, 2, 2, 2], 2, 6, True),
        ('small_gains_do_not_move_anchor', [0, 0.5, 1, 1.5, 2], 2, 4, True),
        ('short_horizon_is_censored', [0, 0, 0], 0, 3, False),
    ]
    for name, values, delta, expected, trigger in cases:
        got, fired, trace = patience(values, delta)
        assert (got, fired) == (expected, trigger), name
        checks.append({'name': name, 'pass': True, 'stop_round': got})
    return checks

def legacy_record(kind, candidate, dataset, eval_rows, previous_hashes):
    matches = [x for x in eval_rows if x['run_id'] == OLD_RUN and x['phase'] == kind
               and x['candidate'] == candidate and x['dataset'] == dataset]
    if len(matches) != 1:
        raise ValueError(f'Missing/duplicate fixed legacy record: {kind}/{candidate}/{dataset}')
    entry = matches[0]
    path = ROOT / entry['result_source']
    value = js(path)
    if INPUTS[str(path.relative_to(ROOT))] != previous_hashes[entry['result_source']]:
        raise ValueError('Legacy input changed since E1')
    for field in ('correct', 'total', 'seed', 'model', 'accuracy'):
        assert str(value[field]) == entry[field], (entry['result_source'], field)
    assert value['accuracy'] == value['correct'] / value['total']
    assert value['seed'] == 42 and value['model'] == 'openrouter/openai/gpt-oss-120b'
    return entry, value

def main():
    boundary_checks = rule_checks()
    all_rounds = rows(AUDIT / 'rounds.csv')
    candidate_rows = rows(AUDIT / 'candidates.csv')
    eval_rows = rows(AUDIT / 'evaluations.csv')
    previous_hashes = js(AUDIT / 'source_hashes.json')
    legacy = sorted((x for x in all_rounds if x['group'] == 'text_classification_reproduction'
                     and int(x['round']) <= HORIZON), key=lambda x: int(x['round']))
    assert [int(x['round']) for x in legacy] == list(range(1, HORIZON + 1))
    assert all('missing' not in x['incumbent_after'] for x in legacy)
    attempted = [x for x in candidate_rows if x['run_id'] == OLD_RUN and int(x['round']) <= HORIZON]
    assert len(attempted) == 38
    assert all(sum(int(x['round']) == t for x in attempted) == 2 for t in range(1, 20))
    repaired = {x['candidate'] for x in attempted if x['recovered_after_search'] == 'True'}
    assert repaired == {'hard_buffer_memory', 'per_label_recent_memory'}
    assert not repaired.intersection(x['incumbent_after'] for x in legacy)
    definition = read(ROOT / 'reference_examples/text_classification/benchmark.py')
    assert 'avg_acc = sum(stats["accs"]) / len(stats["accs"])' in definition

    validation, exact_validation, provenance = {}, {}, []
    for candidate in dict.fromkeys(x['incumbent_after'] for x in legacy):
        parts, exact_parts = [], []
        for ds in DATASETS:
            entry, value = legacy_record('val', candidate, ds, eval_rows, previous_hashes)
            parts.append(float(entry['accuracy']) * 100)
            exact_parts.append(Fraction(int(entry['correct']), int(entry['total'])) * 100)
            provenance.append({'candidate': candidate, 'dataset': ds, 'kind': 'val',
                               'source': entry['result_source'], 'accuracy': value['accuracy'],
                               'correct': value['correct'], 'total': value['total'],
                               'timestamp': value['timestamp'], 'sha256': INPUTS[entry['result_source']]})
        validation[candidate] = sum(parts) / len(parts)
        exact_validation[candidate] = sum(exact_parts, Fraction()) / len(exact_parts)
    values = [validation[x['incumbent_after']] for x in legacy]
    exact_values = [exact_validation[x['incumbent_after']] for x in legacy]
    p0, f0, trace0 = patience(values, 0)
    p2, f2, trace2 = patience(values, 2)
    exact0, ef0, exact_trace0 = patience(exact_values, Fraction(0), True)
    exact2, ef2, exact_trace2 = patience(exact_values, Fraction(2), True)
    assert (p0, f0, p2, f2) == (exact0, ef0, exact2, ef2), 'Precision-dependent stop; uncertain'
    for machine, exact in ((trace0, exact_trace0), (trace2, exact_trace2)):
        assert len(machine) == len(exact)
        assert all(all(m[k] == e[k] for k in ('round','reset','count','stop')) for m,e in zip(machine,exact))
    # Freeze all four choices before fetching any selected test record.
    selections = [('Reference-19', 19, 'reference_horizon'), ('Fixed-10', 10, 'fixed_budget'),
                  ('Patience-3 delta0', p0, 'triggered' if f0 else 'not_triggered_by_19'),
                  ('Patience-3 delta2pp', p2, 'triggered' if f2 else 'not_triggered_by_19')]
    save('frozen_validation_selections.json', [
        {'policy': name, 'stop_round': t, 'incumbent': legacy[t-1]['incumbent_after'], 'status': status}
        for name, t, status in selections])
    tests, exact_tests = {}, {}
    for candidate in dict.fromkeys(legacy[t-1]['incumbent_after'] for _, t, _ in selections):
        tests[candidate], exact_tests[candidate] = {}, {}
        for ds in DATASETS:
            entry, value = legacy_record('test', candidate, ds, eval_rows, previous_hashes)
            tests[candidate][ds] = float(entry['accuracy']) * 100
            exact_tests[candidate][ds] = Fraction(int(entry['correct']), int(entry['total'])) * 100
            provenance.append({'candidate': candidate, 'dataset': ds, 'kind': 'test',
                               'source': entry['result_source'], 'accuracy': value['accuracy'],
                               'correct': value['correct'], 'total': value['total'],
                               'timestamp': value['timestamp'], 'sha256': INPUTS[entry['result_source']]})
    ref_candidate = legacy[-1]['incumbent_after']
    comparison, loss_flags = [], []
    for name, t, status in selections:
        candidate = legacy[t-1]['incumbent_after']
        count = sum(int(x['new_candidates']) for x in legacy[:t])
        item = {'policy': name, 'stop_round': t, 'incumbent': candidate,
                'validation_pp': validation[candidate], 'validation_exact_count_fraction_pp': str(exact_validation[candidate]),
                'validation_delta_vs_reference_pp': float(exact_validation[candidate]-exact_validation[ref_candidate]),
                'trigger_status': status, 'attempted_candidates': count, 'fewer_candidates_vs_reference19': 38-count,
                'token_savings': 'missing_no_comparable_ledger', 'usd_savings': 'missing_no_comparable_ledger',
                'time_savings': 'missing_no_comparable_ledger', 'interpretation': 'conditional_legacy_summary_comparison'}
        for ds in DATASETS:
            delta = exact_tests[candidate][ds] - exact_tests[ref_candidate][ds]
            item[ds+'_test_pp'] = tests[candidate][ds]
            item[ds+'_delta_vs_reference_pp'] = float(delta)
            flag = {'policy': name, 'dataset': ds, 'delta_pp_exact_fraction': str(delta), 'delta_pp': float(delta)}
            flag.update({f'lower_by_ge_{k}pp': delta <= -k for k in (1, 2, 5)})
            loss_flags.append(flag)
        comparison.append(item)

    short, pair_signatures = [], {}
    for row in all_rounds:
        if row['group'] not in ('lawbench_pilot', 'lawbench_injected'): continue
        cp_path = ROOT / row['checkpoint_evidence']; cp = js(cp_path)
        valid = [x for x in cp['candidates'] if x.get('correct') is not None]
        assert cp['selected'] == max(valid, key=lambda x: x['correct'])['candidate']
        selected = next(x for x in valid if x['candidate'] == cp['selected'])
        item = {'group': row['group'], 'run_id': row['run_id'], 'round': cp['T'],
                'D': cp['D'], 'incumbent': cp['selected'], 'score_pp': 100*selected['correct']/selected['total'],
                'checkpoint_time_utc': cp['created_at_utc'], 'selection_source': row['checkpoint_evidence']}
        if cp['selected'] == BASE:
            assert row['group'] == 'lawbench_injected'
            item.update({'audit_pp': 27.5, 'audit_basis': 'shared_H0_existing_two_repeats',
                         'audit_repeat_pp': '29;26', 'new_independent_audit': False})
            h0 = [js(ROOT / 'lawbench_rsi_v3/external/noise/R4B' / str(i) / 'audit/result.json') for i in (1, 2)]
            assert [x['correct'] for x in h0] == [29, 26] and all(x['total'] == 100 for x in h0)
        else:
            er = next(x for x in eval_rows if x['run_id'] == row['run_id'] and x['candidate'] == cp['selected'] and x['phase'] == 'audit')
            path = ROOT / er['result_source']; audit = js(path); done = js(path.parent / 'complete.json')
            assert INPUTS[str(path.relative_to(ROOT))] == done['files']['result.json']
            item.update({'audit_pp': audit['accuracy']*100, 'audit_basis': 'existing_fixed_incumbent_audit',
                         'audit_measurement_round': audit['round'], 'audit_source': er['result_source'],
                         'new_independent_audit': False})
            if row['group'] == 'lawbench_pilot':
                pair_signatures[(row['run_id'], cp['T'])] = {'seed': audit['seed'], 'config': audit['model_config'],
                    'item_ids_hash': done['signature']['item_ids_hash'], 'manifest_hash': done['signature']['manifest_hash'],
                    'total': audit['total'], 'complete_source': str((path.parent/'complete.json').relative_to(ROOT))}
        short.append(item)
    declared_pairs = rows(ROOT/'lawbench_rsi_v3/analysis/paired_changes.csv')
    pairs = []
    for suffix in ('a', 'b'):
        for t in (1, 2):
            keys = (('D0_'+suffix, t), ('D100_'+suffix, t))
            sig0, sig100 = (pair_signatures[k] for k in keys)
            comparable = all(sig0[k] == sig100[k] for k in ('seed','config','item_ids_hash','manifest_hash','total'))
            left, right = (next(x for x in short if (x['run_id'],x['round']) == k) for k in keys)
            declared = next(x for x in declared_pairs if x['axis']=='D' and x['comparison']==f'repeat={suffix},T={t}')
            assert declared['before'] == left['run_id']+'/'+left['incumbent']
            assert declared['after'] == right['run_id']+'/'+right['incumbent']
            pairs.append({'trajectory_pair': suffix, 'round': t, 'D0_run': keys[0][0], 'D100_run': keys[1][0],
                          'pairing_confirmed': comparable, 'audit_items_per_pair': sig0['total'] if comparable else 'missing',
                          'D100_minus_D0_audit_pp': right['audit_pp']-left['audit_pp'] if comparable else 'missing',
                          'seed': sig0['seed'], 'audit_input_ids_hash': sig0['item_ids_hash'],
                          'causal_dose_interpretation': 'not_identified; exposure_protocol_only'})

    exposure_path = ROOT / 'lawbench_rsi_v3/analysis/optimizer_evidence_access.csv'
    exposure = [x for x in rows(exposure_path) if x['phase'] == 'feedback']
    direct = []
    for run in ('D100_a', 'D100_b'):
        for t in (1, 2):
            p = ROOT/'lawbench_rsi_v3/external/control'/run/f'proposer_round{t}/feedback_access.json'
            rec = js(p); calls = rec.get('observed_feedback_reads', [])
            visible = set(i for c in calls if not c.get('is_error') for i in c.get('item_ids_in_tool_output', []))
            direct.append({'run_id':run, 'round':t, 'observed_feedback_read_events':len(calls),
                           'failed_read_events':sum(bool(c.get('is_error')) for c in calls),
                           'distinct_returned_item_ids':len(visible), 'actual_cognitive_use':'unknown', 'source':str(p.relative_to(ROOT))})
    injected = js(ROOT/'lawbench_rsi_v3/analysis/injected_results.json')

    registry, alias = [], {'fewshot_all':'B0'}
    for i, c in enumerate((x for x in candidate_rows if x['run_id']==OLD_RUN), 1):
        key = f'C{i:02d}'; alias[c['candidate']] = key
        registry.append({'alias':key, 'round':int(c['round']), 'candidate':c['candidate'],
                         'raw_recorded_score_pp':c['raw_validation_pp'], 'status':c['evaluation_status'],
                         'recovered_after_search':c['recovered_after_search']})
    registry.insert(0, {'alias':'B0', 'round':0, 'candidate':'fewshot_all', 'status':'baseline'})
    # Plot data uses only E1 exported rows. No additional project retrieval here.
    plot = []
    for row in sorted((x for x in all_rounds if x['run_id']==OLD_RUN), key=lambda x:int(x['round'])):
        t = int(row['round']); cr = [x for x in candidate_rows if x['run_id']==OLD_RUN and int(x['round'])==t]
        out = {'t':t}
        for k,c in zip(('a','b'),cr):
            valid = c['evaluation_status']=='recorded_evaluation'
            out[k+'_id']=alias[c['candidate']]; out[k+'_pp']=float(c['raw_validation_pp']) if valid else ''
            out[k+'_status']=c['evaluation_status']; out[k+'_runtime_zero_if_failed']=0 if c['evaluation_status']=='benchmark_failed_logged_zero' else ''
        known=t<=HORIZON; name=row['incumbent_after']
        out.update({'inc_id':alias[name] if known else '', 'V_pp':validation[name] if known else '',
                    'within_comparison_horizon':known, 'unknown_incumbent':not known})
        for ds, col in zip(DATASETS,('U_pp','S_pp','L_pp')):
            out[col]=float(row['fixed_global_incumbent_'+ds+'_test_accuracy'])*100 if known else ''
        plot.append(out)

    for filename, data in [('comparison.csv',comparison), ('descriptive_loss_flags.csv',loss_flags),
        ('patience_trace.csv',[{'policy':'delta0',**x} for x in trace0]+[{'policy':'delta2pp',**x} for x in trace2]),
        ('lawbench_selected.csv',short), ('lawbench_pairs.csv',pairs), ('feedback_available_and_returned.csv',exposure),
        ('feedback_direct_read_evidence.csv',direct), ('plot_rounds.csv',plot), ('candidate_name_registry.csv',registry),
        ('selected_score_sources.csv',provenance), ('original_failure_status.csv',[x for x in attempted if x['evaluation_status']!='recorded_evaluation'])]:
        write_csv(filename,data)
    save('protocol.json', {'horizon':19, 'actual_old_round20_incumbent':'missing', 'fixed_budget':10,
        'patience':3, 'initialization_round':1, 'initial_count':0, 'delta0_reset':'V_t > anchor',
        'delta2pp_reset':'V_t - anchor >= 2; small gains accumulate relative to unchanged anchor',
        'opportunity_unit':'whole round containing two candidates', 'equal_score_id_change_resets':False,
        'policy_choice_uses_test':False, 'new_experimental_model_calls':0, 'precision':'original machine-readable accuracy; no rounding before decisions',
        'plot_source':'E1 exported CSV only; candidate raw scores retain original one-decimal records; failures separate'})
    save('pairing_and_exposure_evidence.json', {'distinct_trajectory_pairs':2, 'roundwise_contrasts':4,
        'not_four_independent_pairs':True, 'pair_signatures':{str(k):v for k,v in pair_signatures.items()},
        'direct_feedback_read_counts':direct, 'injected_inline_count_existing_verification':injected['feedback_records_inline'],
        'injected_feedback_sha256':injected['feedback_sha256'], 'injected_prompt_sha256':injected['feedback_prompt_utf8_sha256'],
        'actual_use_of_exposed_information':'unknown', 'R4B_R7A_R12B_materials':'existing twice-measured versions; E3 not analyzed'})
    changed=[p for p,sha in INPUTS.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha]
    assert not changed
    save('input_manifest.json', INPUTS)
    save('checks.json', {'rule_boundary_checks':boundary_checks, 'native_float_and_exact_count_rules_agree':True,
        'all_used_reset_counter_decisions_agree_across_precision_checks':True,
        'round1_count_zero':True, 'candidate_opportunities_not_counted_twice':True, 'after_stop_not_used_for_decision':True,
        'failed_or_recovered_candidates_not_used_as_incumbents':True, 'strict_round20_identity_left_missing':True,
        'selection_saved_before_test_lookup':True, 'paired_audit_config_checks':all(x['pairing_confirmed'] for x in pairs),
        'input_files':len(INPUTS), 'changed_inputs':changed, 'new_experimental_model_calls':0,
        'pair_labels_confirmed_by_existing_paired_changes_log':True,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({'comparison':[{k:x[k] for k in ('policy','stop_round','incumbent','validation_pp','attempted_candidates','fewer_candidates_vs_reference19')+tuple(ds+s for ds in DATASETS for s in ('_test_pp','_delta_vs_reference_pp'))} for x in comparison],
        'lawbench_pairs':[{k:x[k] for k in ('trajectory_pair','round','pairing_confirmed','audit_items_per_pair','D100_minus_D0_audit_pp')} for x in pairs],
        'feedback_reads':[{k:x[k] for k in ('run_id','round','observed_feedback_read_events','failed_read_events','distinct_returned_item_ids')} for x in direct],
        'checks':'PASS', 'input_files':len(INPUTS), 'new_experimental_model_calls':0},ensure_ascii=False))

if __name__ == '__main__':
    main()
