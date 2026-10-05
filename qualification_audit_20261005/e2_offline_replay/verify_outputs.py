"""Independent checks of generated E2 artifacts; no project execution."""
from pathlib import Path
from fractions import Fraction
import ast, csv, hashlib, json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def csv_rows(name):
    with (HERE/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

main = csv_rows('comparison.csv')
plot = csv_rows('plot_rounds.csv')
trace = csv_rows('patience_trace.csv')
flags = csv_rows('descriptive_loss_flags.csv')
assert len(main)==4 and len(plot)==20 and len(flags)==12
assert [int(x['t']) for x in plot]==list(range(1,21))
assert all(not plot[-1][k] for k in ('inc_id','V_pp','U_pp','S_pp','L_pp'))
for row in main:
    t=int(row['stop_round'])
    assert int(row['attempted_candidates'])==2*t
    assert int(row['fewer_candidates_vs_reference19'])==38-2*t
    assert float(row['validation_pp'])==float(plot[t-1]['V_pp'])
    for ds,col in [('USPTO','U_pp'),('Symptom2Disease','S_pp'),('LawBench','L_pp')]:
        assert float(row[ds+'_test_pp'])==float(plot[t-1][col])
    assert all(row[k]=='missing_no_comparable_ledger' for k in ('token_savings','usd_savings','time_savings'))
for policy, delta in [('delta0',Fraction(0)),('delta2pp',Fraction(2))]:
    tr=[x for x in trace if x['policy']==policy]
    assert int(tr[0]['round'])==1 and int(tr[0]['count'])==0
    vals=[Fraction(x['V_pp']) for x in plot[:19]]
    reset_positions=[0]
    expected_end=19
    for i in range(1,19):
        if (vals[i]>vals[reset_positions[-1]]) if delta==0 else (vals[i]-vals[reset_positions[-1]]>=delta):
            reset_positions.append(i)
        if i-reset_positions[-1]==3:
            expected_end=i+1
            break
    assert len(tr)==expected_end
    assert int(tr[-1]['round'])==expected_end and tr[-1]['stop']=='True'
    assert int(tr[-1]['count'])==3
for x in flags:
    difference=Fraction(x['delta_pp_exact_fraction'])
    for k in (1,2,5):
        assert (x[f'lower_by_ge_{k}pp']=='True')==(difference<=-k)
failed=csv_rows('original_failure_status.csv')
assert {(x['candidate'],x['evaluation_status']) for x in failed}=={
    ('hard_buffer_memory','benchmark_failed_logged_zero'),('per_label_recent_memory','import_failed_no_score')}
assert plot[15]['a_pp']=='' and plot[15]['a_runtime_zero_if_failed']=='0'
assert plot[18]['b_pp']=='' and plot[18]['b_status']=='import_failed_no_score'
imports=set()
tree=ast.parse((HERE/'replay.py').read_text(encoding='utf-8'))
for node in ast.walk(tree):
    if isinstance(node,ast.Import): imports.update(x.name.split('.')[0] for x in node.names)
    if isinstance(node,ast.ImportFrom): imports.add(node.module.split('.')[0])
assert imports<={'pathlib','fractions','csv','hashlib','io','json'}
manifest=json.loads((HERE/'input_manifest.json').read_text(encoding='utf-8'))
changed=[p for p,h in manifest.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
assert not changed
check=json.loads((HERE/'checks.json').read_text(encoding='utf-8'))
assert check['script_sha256']==hashlib.sha256((HERE/'replay.py').read_bytes()).hexdigest()
result={'status':'PASS','independent_output_checks':True,'input_files':len(manifest),'changed_inputs':changed,
        'script_standard_library_only':True,'four_frozen_policy_rows':4,'round20_unknown_preserved':True,
        'new_experimental_model_calls':0,'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(HERE/'output_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
