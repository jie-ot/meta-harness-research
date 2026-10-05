"""One-time pre-live repair of Windows-translated prompt files.

The lock hashes were calculated from the correct in-memory strings and already
matched the historical R4B trace. Re-materialize those exact strings as UTF-8
bytes without newline translation. No model request is made.
"""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TRACE = ROOT / 'lawbench_rsi_v3/external/noise/R4B/1/score/score_traces.jsonl'
items = [json.loads(line) for line in (OUT / 'locked_items.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
if (OUT / 'live_request_attempted.json').exists():
    raise SystemExit('A live request marker exists; prompt repair is forbidden.')
historical = []
with TRACE.open(encoding='utf-8') as handle:
    for _ in range(20):
        historical.append(json.loads(next(handle)))
before, after = [], []
for item, trace in zip(items, historical):
    if trace['item_id'] != item['item_id']:
        raise ValueError('Historical item order changed.')
    prompt = trace['prompt_text']
    exact_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    if exact_hash != item['prompt_sha256'] or trace['constructed_prompt_text'] != prompt:
        raise ValueError('Historical prompt no longer matches the pre-live lock.')
    path = OUT / item['prompt_path']
    before.append(hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_bytes(prompt.encode('utf-8'))
    after.append(hashlib.sha256(path.read_bytes()).hexdigest())
if after != [x['prompt_sha256'] for x in items]:
    raise ValueError('Byte-preserving rematerialization failed.')
record = {'status': 'PASS', 'when': 'before any live request',
          'reason': 'Path.write_text translated mixed LawBench newlines on Windows; lock hashes and historical strings were correct',
          'files_rewritten_in_new_e4_directory': 20,
          'before_physical_hashes_sha256': hashlib.sha256(json.dumps(before).encode()).hexdigest(),
          'after_hashes_equal_locked_prompt_hashes': True,
          'source_files_modified': False, 'new_model_requests': 0}
(OUT / 'prompt_byte_repair.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, ensure_ascii=False))
