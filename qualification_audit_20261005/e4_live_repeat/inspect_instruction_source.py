"""Inspect discovered instruction-source provenance without exposing its text."""
import datetime
import hashlib
import json
import queue
import subprocess
import threading
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
NODE = r'D:\DevEnvs\Nodejs\node.exe'
CLI = r'D:\OpenAI\CodexBin\node_modules\@openai\codex\bin\codex.js'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    destination = OUT / 'instruction_source_assessment.json'
    if destination.exists():
        raise SystemExit('Instruction source was already assessed; no reconnect.')
    request = json.loads((OUT / 'request_config.json').read_text(encoding='utf-8'))
    locked = [json.loads(x) for x in (OUT / 'locked_items.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    prompt = (OUT / locked[0]['prompt_path']).read_bytes().decode('utf-8')
    markers = ['meta-harness', 'lawbench', 'r4b', 'confusion_disambiguation_memory',
               'gpt-oss-120b', 'qualification_audit', locked[0]['prompt_sha256']]
    markers += [x['item_id'] for x in locked]
    markers += [x['target'] for x in locked if len(x['target']) >= 4]
    # Long excerpts distinguish task data without storing the excerpts.
    markers += [prompt[0:120], prompt[-120:]]
    proc = subprocess.Popen([NODE, CLI, 'app-server', '--stdio'], cwd=str(OUT),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, encoding='utf-8', errors='replace', bufsize=1,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    messages = queue.Queue()

    def reader():
        try:
            for line in proc.stdout:
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(value, dict):
                    messages.put(value)
        finally:
            messages.put(None)

    threading.Thread(target=reader, daemon=True).start()
    counter = 0

    def rpc(method, params):
        nonlocal counter
        counter += 1
        proc.stdin.write(json.dumps({'id': counter, 'method': method, 'params': params}, ensure_ascii=False) + '\n')
        proc.stdin.flush()
        while True:
            value = messages.get(timeout=20)
            if value is None:
                raise RuntimeError('stdio_exited')
            if value.get('id') != counter:
                continue
            if 'error' in value:
                error = value['error'] if isinstance(value['error'], dict) else {}
                raise RuntimeError('rpc_error_' + str(error.get('code')))
            return value['result']

    cleanup = 'unknown'
    try:
        rpc('initialize', {'clientInfo': {'name': 'meta_harness_e4_instruction_audit',
             'title': 'E4 instruction-source audit', 'version': '1.0.0'},
             'capabilities': {'experimentalApi': False}})
        proc.stdin.write('{"method":"initialized"}\n')
        proc.stdin.flush()
        response = rpc('thread/start', request['thread_start'])
        sources = response.get('instructionSources') if isinstance(response.get('instructionSources'), list) else []
        rows = []
        for source in sources:
            if not isinstance(source, str):
                rows.append({'source_type': 'non_path_or_unknown', 'fixable_by_hash': False,
                             'task_data_detected': 'unknown', 'reason': 'source entry was not a string'})
                continue
            source_path = Path(source)
            normalized = str(source_path.resolve()) if source_path.is_absolute() else source
            source_type = 'external_or_unknown'
            try:
                source_path.resolve().relative_to(ROOT.resolve())
                source_type = 'project_tree'
            except (ValueError, OSError):
                lower = normalized.casefold()
                if '\\openai\\codexhome\\' in lower:
                    source_type = 'codex_application_home'
                elif '\\.codex\\' in lower or '/.codex/' in lower:
                    source_type = 'user_global_codex'
            row = {'source_type': source_type, 'path_sha256': digest(normalized.encode('utf-8')),
                   'basename': source_path.name, 'exists_as_file': source_path.is_file(),
                   'fixable_by_hash': False, 'task_data_detected': 'unknown',
                   'personalization_marker_detected': 'unknown'}
            if source_path.is_file():
                data = source_path.read_bytes()
                row.update({'content_sha256': digest(data), 'bytes': len(data), 'fixable_by_hash': True})
                try:
                    text = data.decode('utf-8')
                    folded = text.casefold()
                    matched = [marker for marker in markers if marker and marker.casefold() in folded]
                    personal_markers = ['writing style', 'user preference', 'personal preference',
                                        '用户偏好', '写作风格', 'personal memory']
                    row['task_data_detected'] = bool(matched)
                    row['task_marker_match_count'] = len(matched)
                    row['personalization_marker_detected'] = any(x in folded for x in personal_markers)
                    row['contains_known_item_ids'] = any(x['item_id'].casefold() in folded for x in locked)
                    row['contains_known_targets'] = any(len(x['target']) >= 4 and x['target'].casefold() in folded for x in locked)
                    row['contains_prompt_excerpt'] = prompt[:120].casefold() in folded or prompt[-120:].casefold() in folded
                except UnicodeDecodeError:
                    row['reason'] = 'not_utf8; content semantic scan unavailable'
            rows.append(row)
        acceptable = bool(rows) and all(x.get('fixable_by_hash') is True and x.get('task_data_detected') is False
                                        and x.get('source_type') != 'project_tree' for x in rows)
        report = {'queried_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'instruction_source_count': len(rows), 'sources': rows,
                  'raw_paths_or_contents_saved': False,
                  'thread_echo': {'model': response.get('model'), 'reasoning_effort': response.get('reasoningEffort'),
                                  'ephemeral': response.get('thread', {}).get('ephemeral')},
                  'accept_as_fixed_wrapper': acceptable,
                  'acceptance_rule': 'each source is outside project tree, byte-hashable, and contains none of the locked task markers/IDs/targets/prompt excerpts',
                  'inference_turns_sent': 0}
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=4)
            cleanup = 'this_ephemeral_helper_exited'
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=8, creationflags=subprocess.CREATE_NO_WINDOW)
            cleanup = 'only_this_helper_tree_terminated'
    report['helper_cleanup'] = cleanup
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'instruction_source_count': report['instruction_source_count'],
                      'sources': report['sources'],
                      'accept_as_fixed_wrapper': report['accept_as_fixed_wrapper'],
                      'thread_echo': report['thread_echo'], 'helper_cleanup': cleanup}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
