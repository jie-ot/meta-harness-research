"""Run exactly one locked E4 observation through official Codex app-server.

Uses the existing ChatGPT login. It makes one turn/start and never retries it.
No credential files are read, no settings are persisted, and no project module
is imported. Any tool item makes the observation invalid and triggers interrupt.
"""
import datetime
import hashlib
import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
OBS = OUT / 'observation_000_live'
ROOT = OUT.parents[1]
NODE = r'D:\DevEnvs\Nodejs\node.exe'
CLI = r'D:\OpenAI\CodexBin\node_modules\@openai\codex\bin\codex.js'
TARGET_MODEL = 'gpt-5.6-sol'
TARGET_EFFORT = 'max'
EXPECTED_CONFIG_HASH = '3dacf214af4260a058ff025f5d7b096e9a9a6b4b3abd3e9f864c006a8b55614b'
EXPECTED_CONTINUATION_HASH = '687a79410d82257a03f7977242299149b380e692566be4c2edf3e53da619f030'
MAX_SECONDS = 300
TOOL_TYPES = {'commandExecution', 'fileChange', 'mcpToolCall', 'dynamicToolCall',
              'collabAgentToolCall', 'webSearch', 'imageView', 'imageGeneration'}
DISABLED_FEATURES = ['shell_tool', 'unified_exec', 'shell_snapshot', 'memories',
                     'apps', 'browser_use', 'computer_use', 'image_generation',
                     'multi_agent', 'plugins', 'view_image']


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def text_sha(text):
    return sha(text.encode('utf-8'))


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def epoch_utc(value):
    if type(value) is not int:
        return None
    try:
        return datetime.datetime.fromtimestamp(value, datetime.timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def window(value):
    if not isinstance(value, dict):
        return None
    used = value.get('usedPercent')
    return {'window_duration_minutes': value.get('windowDurationMins'),
            'window_kind': 'weekly_7_days' if value.get('windowDurationMins') == 10080 else 'other_or_unknown',
            'used_percent': used,
            'remaining_percent_derived': 100 - used if type(used) is int and 0 <= used <= 100 else None,
            'resets_at_epoch_seconds': value.get('resetsAt'),
            'resets_at_utc': epoch_utc(value.get('resetsAt'))}


def rate_snapshot(response, queried, received):
    rows = []
    by_id = response.get('rateLimitsByLimitId')
    if isinstance(by_id, dict):
        for key, value in by_id.items():
            if isinstance(value, dict):
                rows.append({'bucket_map_key': key, 'limit_id': value.get('limitId'),
                             'limit_name': value.get('limitName'), 'primary': window(value.get('primary')),
                             'secondary': window(value.get('secondary'))})
    legacy = response.get('rateLimits')
    if isinstance(legacy, dict) and not any(x['limit_id'] == legacy.get('limitId') for x in rows):
        rows.append({'bucket_map_key': None, 'limit_id': legacy.get('limitId'),
                     'limit_name': legacy.get('limitName'), 'primary': window(legacy.get('primary')),
                     'secondary': window(legacy.get('secondary'))})
    return {'queried_at_utc': queried, 'received_at_utc': received, 'buckets': rows,
            'model_bucket_mapping': 'unknown: model catalog has no rate-limit bucket mapping'}


def extract_json_field(text, field, default=''):
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return str(value.get(field, default))
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r'```(?:json)?\s*([\s\S]*?)\s*```', text):
        try:
            value = json.loads(match.group(1))
            if isinstance(value, dict):
                return str(value.get(field, default))
        except json.JSONDecodeError:
            pass
    for start in range(len(text)):
        if text[start] != '{':
            continue
        depth, pos, in_string = 1, start + 1, False
        while pos < len(text) and depth > 0:
            char = text[pos]
            if char == '"' and (pos == 0 or text[pos - 1] != '\\'):
                in_string = not in_string
            elif not in_string:
                depth += 1 if char == '{' else (-1 if char == '}' else 0)
            pos += 1
        if depth == 0:
            candidate = re.sub(r',\s*([\]}])', r'\1', text[start:pos])
            try:
                value = json.loads(candidate)
                if isinstance(value, dict):
                    return str(value.get(field, default))
            except json.JSONDecodeError:
                pass
    match = re.findall(rf'"{field}"\s*:\s*"([^"]*)"', text)
    return match[-1] if match else default


def canonical(text):
    text = text.strip()
    match = re.search(r'\[罪名\](.*?)(?:<eoa>|$)', text)
    if match:
        text = match.group(1).strip()
    elif '罪名:' in text:
        text = text.split('罪名:')[-1]
    text = re.sub(r'<eoa>.*', '', text).strip()
    for separator in [';', '；', ',', '，', '、']:
        if separator in text:
            return sorted({part.strip() for part in text.split(separator) if part.strip()})
    return [text] if text else []


class ProtocolStop(Exception):
    def __init__(self, kind, method=None, code=None):
        self.kind, self.method, self.code = kind, method, code


def main():
    continuation_marker = OUT / 'live_request_continuation_attempted.json'
    if continuation_marker.exists():
        raise SystemExit('The one authorized actual turn was already attempted; no retry.')
    prior = json.loads((OUT / 'observation_000/result.json').read_text(encoding='utf-8'))
    if (prior.get('turn_start_requests_sent') != 0
            or prior.get('blocker', {}).get('classification') != 'thread_isolation_not_confirmed'):
        raise SystemExit('Prior gate was not a zero-turn instruction-source stop; continuation forbidden.')
    assessment = json.loads((OUT / 'instruction_source_assessment.json').read_text(encoding='utf-8'))
    continuation_bytes = (OUT / 'continuation_config.json').read_bytes()
    continuation = json.loads(continuation_bytes)
    if (sha(continuation_bytes) != EXPECTED_CONTINUATION_HASH
            or assessment.get('accept_as_fixed_wrapper') is not True):
        raise SystemExit('Continuation or instruction-source assessment changed; do not send.')
    OBS.mkdir(parents=True, exist_ok=True)
    lock = json.loads((OUT / 'lock_manifest.json').read_text(encoding='utf-8'))
    config_bytes = (OUT / 'request_config.json').read_bytes()
    config = json.loads(config_bytes)
    if lock['new_model_requests_so_far'] != 0 or sha(config_bytes) != lock['request_config_sha256'] != EXPECTED_CONFIG_HASH:
        raise SystemExit('Locked request configuration changed; do not send.')
    items = [json.loads(line) for line in (OUT / 'locked_items.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(items) != 20 or [x['item_id'] for x in items] != lock['selected_item_ids']:
        raise SystemExit('Locked item panel changed; do not send.')
    item = items[0]
    prompt_path = OUT / item['prompt_path']
    # Decode bytes directly so Python does not normalize embedded CRLF from the
    # original LawBench cases. The hash is over the exact outgoing string.
    prompt = prompt_path.read_bytes().decode('utf-8')
    if text_sha(prompt) != item['prompt_sha256']:
        raise SystemExit('First prompt changed; do not send.')
    attempted = {'attempted_at_utc': utcnow(), 'authorized_turn_start_limit': 1,
                 'item_index': 0, 'item_id': item['item_id'],
                 'prompt_sha256': item['prompt_sha256'], 'prompt_chars': len(prompt),
                 'model': TARGET_MODEL, 'effort': TARGET_EFFORT,
                 'answer_retry_policy': 'never retry this observation'}
    attempted['prior_zero_turn_gate_result'] = 'observation_000/result.json'
    attempted['continuation_config_sha256'] = sha(continuation_bytes)
    atomic_json(continuation_marker, attempted)
    result = {'status': 'preparing_connection', 'attempt': attempted,
              'request_config_sha256': sha(config_bytes), 'turn_start_requests_sent': 0,
              'rate_limits_before': None, 'rate_limits_after': None,
              'catalog_evidence': None, 'model_echo': None, 'token_usage': None,
              'tool_item_types': [], 'response': None, 'blocker': None,
              'client_result_cache': 'none used; old harness cache bypassed; provider input cache reported separately',
              'semantic_differences': config['semantic_difference'],
              'temporary_disabled_features': DISABLED_FEATURES,
              'instruction_source_assessment': 'fixed hash, no locked task markers; see instruction_source_assessment.json',
              'persistent_settings_changed': False, 'authentication_files_read': False}
    atomic_json(OBS / 'progress.json', result)
    proc = None
    cleanup = 'not_started'
    notifications = []
    agent_messages = []
    item_types = []
    token_usage = None
    settings_echo = None
    turn_complete = None
    tool_violation = False
    request_id = 0
    thread_id = None
    turn_id = None
    turn_started_monotonic = None
    turn_completed_monotonic = None
    try:
        deadline = time.monotonic() + MAX_SECONDS
        command = [NODE, CLI, 'app-server', '--stdio']
        for feature in DISABLED_FEATURES:
            command.extend(['--disable', feature])
        proc = subprocess.Popen(command, cwd=str(OUT),
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

        def send(method, params=None, include_params=True):
            nonlocal request_id
            request_id += 1
            value = {'id': request_id, 'method': method}
            if include_params:
                value['params'] = params
            proc.stdin.write(json.dumps(value, ensure_ascii=False) + '\n')
            proc.stdin.flush()
            return request_id

        def notify(method, params=None):
            value = {'method': method}
            if params is not None:
                value['params'] = params
            proc.stdin.write(json.dumps(value, ensure_ascii=False) + '\n')
            proc.stdin.flush()

        def handle(value):
            nonlocal token_usage, settings_echo, turn_complete, tool_violation, turn_id, turn_completed_monotonic
            method = value.get('method')
            if not isinstance(method, str):
                return
            params = value.get('params') if isinstance(value.get('params'), dict) else {}
            entry = {'method': method, 'received_at_utc': utcnow()}
            if method in ('item/started', 'item/completed'):
                item_value = params.get('item') if isinstance(params.get('item'), dict) else {}
                kind = item_value.get('type')
                entry['item_type'] = kind
                item_types.append(kind)
                if kind == 'agentMessage' and method == 'item/completed':
                    agent_messages.append({'id': item_value.get('id'), 'phase': item_value.get('phase'),
                                           'text': item_value.get('text', '')})
                if kind in TOOL_TYPES:
                    tool_violation = True
                    entry['protocol_violation'] = True
                    if thread_id and (turn_id or params.get('turnId')):
                        send('turn/interrupt', {'threadId': thread_id, 'turnId': turn_id or params.get('turnId')})
            elif method == 'thread/tokenUsage/updated':
                token_usage = params.get('tokenUsage')
                entry['usage_present'] = isinstance(token_usage, dict)
            elif method == 'thread/settings/updated':
                settings = params.get('threadSettings') if isinstance(params.get('threadSettings'), dict) else {}
                settings_echo = {k: settings.get(k) for k in ('model', 'modelProvider', 'effort', 'summary',
                                                               'approvalPolicy', 'personality', 'sandboxPolicy', 'cwd')}
                entry['safe_settings'] = settings_echo
            elif method == 'turn/started':
                turn = params.get('turn') if isinstance(params.get('turn'), dict) else {}
                turn_id = turn.get('id') or turn_id
                entry['turn_id'] = turn_id
            elif method == 'turn/completed':
                turn_complete = params.get('turn') if isinstance(params.get('turn'), dict) else {}
                turn_id = turn_complete.get('id') or turn_id
                turn_completed_monotonic = time.monotonic()
                entry['turn_id'] = turn_id
                entry['turn_status'] = turn_complete.get('status')
            notifications.append(entry)
            atomic_json(OBS / 'event_summary.json', notifications)

        def rpc(method, params=None, include_params=True, timeout=20):
            identity = send(method, params, include_params)
            end = min(deadline, time.monotonic() + timeout)
            while True:
                left = end - time.monotonic()
                if left <= 0:
                    raise ProtocolStop('timeout', method)
                try:
                    value = messages.get(timeout=left)
                except queue.Empty:
                    raise ProtocolStop('timeout', method)
                if value is None:
                    raise ProtocolStop('stdio_process_exited', method)
                if 'method' in value and 'id' in value:
                    raise ProtocolStop('server_requested_client_action', method)
                if 'method' in value:
                    handle(value)
                    continue
                if value.get('id') != identity:
                    continue
                if 'error' in value:
                    error = value['error'] if isinstance(value['error'], dict) else {}
                    raise ProtocolStop('json_rpc_error', method, error.get('code'))
                if not isinstance(value.get('result'), dict):
                    raise ProtocolStop('unexpected_result_shape', method)
                return value['result']

        initialized = rpc('initialize', {'clientInfo': config['client_info'],
                                         'capabilities': {'experimentalApi': False}}, timeout=10)
        result['cli_user_agent'] = initialized.get('userAgent')
        notify('initialized')
        before_query = utcnow()
        before = rpc('account/rateLimits/read', include_params=False, timeout=15)
        result['rate_limits_before'] = rate_snapshot(before, before_query, utcnow())
        del before
        catalog = rpc('model/list', {'limit': 100, 'includeHidden': True}, timeout=15)
        entries = catalog.get('data') if isinstance(catalog.get('data'), list) else []
        matches = [x for x in entries if isinstance(x, dict)
                   and (x.get('id') == TARGET_MODEL or x.get('model') == TARGET_MODEL)]
        if len(matches) != 1 or catalog.get('nextCursor') is not None:
            raise ProtocolStop('target_model_not_uniquely_available', 'model/list')
        entry = matches[0]
        efforts = [x.get('reasoningEffort') for x in entry.get('supportedReasoningEfforts', []) if isinstance(x, dict)]
        result['catalog_evidence'] = {'id': entry.get('id'), 'model': entry.get('model'),
                                      'display_name': entry.get('displayName'),
                                      'supported_reasoning_efforts': efforts,
                                      'max_supported': TARGET_EFFORT in efforts,
                                      'catalog_complete': True}
        if TARGET_EFFORT not in efforts:
            raise ProtocolStop('max_effort_not_supported', 'model/list')
        thread_params = dict(config['thread_start'])
        thread = rpc('thread/start', thread_params, timeout=20)
        thread_value = thread.get('thread') if isinstance(thread.get('thread'), dict) else {}
        thread_id = thread_value.get('id')
        instruction_sources = thread.get('instructionSources') if isinstance(thread.get('instructionSources'), list) else []
        result['model_echo'] = {'thread_id': thread_id, 'model': thread.get('model'),
                                'model_provider': thread.get('modelProvider'),
                                'reasoning_effort': thread.get('reasoningEffort'),
                                'ephemeral': thread_value.get('ephemeral'),
                                'instruction_source_count': len(instruction_sources),
                                'approval_policy': thread.get('approvalPolicy'),
                                'sandbox': thread.get('sandbox')}
        current_sources = []
        for source in instruction_sources:
            if not isinstance(source, str):
                current_sources.append({'valid_path': False})
                continue
            source_path = Path(source)
            normalized = str(source_path.resolve()) if source_path.is_absolute() else source
            current_sources.append({'valid_path': source_path.is_file(),
                                    'path_sha256': sha(normalized.encode('utf-8')),
                                    'content_sha256': sha(source_path.read_bytes()) if source_path.is_file() else None,
                                    'bytes': source_path.stat().st_size if source_path.is_file() else None})
        expected_source = continuation['instruction_source_control']
        wrapper_matches = (len(current_sources) == expected_source['count'] == 1
                           and current_sources[0].get('valid_path') is True
                           and current_sources[0].get('path_sha256') == expected_source['path_sha256']
                           and current_sources[0].get('content_sha256') == expected_source['content_sha256']
                           and current_sources[0].get('bytes') == expected_source['bytes'])
        result['wrapper_control'] = {'source_count': len(current_sources),
                                     'path_and_content_hashes_match_assessment': wrapper_matches,
                                     'task_data_detected_in_assessment': expected_source['task_data_detected'],
                                     'personalization_marker_detected_in_assessment': expected_source['personalization_marker_detected'],
                                     'raw_source_paths_or_contents_saved': False}
        atomic_json(OBS / 'progress.json', result)
        if not thread_id or thread.get('model') != TARGET_MODEL or thread.get('reasoningEffort') != TARGET_EFFORT:
            raise ProtocolStop('model_or_effort_not_echoed_exactly', 'thread/start')
        if thread_value.get('ephemeral') is not True or not wrapper_matches:
            raise ProtocolStop('thread_isolation_not_confirmed', 'thread/start')
        turn_params = dict(config['turn_start'])
        turn_params.pop('input_rule', None)
        turn_params.pop('output_schema', None)
        turn_params['threadId'] = thread_id
        turn_params['input'] = [{'type': 'text', 'text': prompt}]
        outgoing_prompt_hash = text_sha(turn_params['input'][0]['text'])
        if outgoing_prompt_hash != item['prompt_sha256'] or len(turn_params['input']) != 1:
            raise ProtocolStop('outgoing_prompt_lock_failed', 'turn/start')
        result['turn_start_requests_sent'] = 1
        result['outgoing_user_input'] = {'items': 1, 'type': 'text',
                                         'prompt_sha256': outgoing_prompt_hash,
                                         'prompt_chars': len(prompt), 'wrapper_added': False}
        turn_started_monotonic = time.monotonic()
        turn_response = rpc('turn/start', turn_params, timeout=25)
        turn_value = turn_response.get('turn') if isinstance(turn_response.get('turn'), dict) else {}
        turn_id = turn_value.get('id') or turn_id
        result['model_echo']['turn_id'] = turn_id
        atomic_json(OBS / 'progress.json', result)
        while turn_complete is None:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ProtocolStop('turn_completion_timeout', 'turn/start')
            try:
                value = messages.get(timeout=min(left, 15))
            except queue.Empty:
                continue
            if value is None:
                raise ProtocolStop('stdio_process_exited_before_turn_complete', 'turn/start')
            if 'method' in value and 'id' in value:
                raise ProtocolStop('server_requested_client_action', 'turn/start')
            if 'method' in value:
                handle(value)
        after_query = utcnow()
        after = rpc('account/rateLimits/read', include_params=False, timeout=15)
        result['rate_limits_after'] = rate_snapshot(after, after_query, utcnow())
        del after
        final_messages = [x for x in agent_messages if x.get('phase') == 'final'] or agent_messages
        response_text = final_messages[-1]['text'] if final_messages else ''
        (OBS / 'response.txt').write_text(response_text, encoding='utf-8')
        atomic_json(OBS / 'agent_messages.json', agent_messages)
        answer = extract_json_field(response_text, 'final_answer')
        predicted = canonical(answer)
        expected = canonical(item['target'])
        turn_status = turn_complete.get('status')
        result['status'] = 'success' if turn_status == 'completed' and response_text and not tool_violation else 'failed'
        result['response'] = {'raw_response_path': 'response.txt', 'raw_response_sha256': text_sha(response_text),
                              'raw_response_chars': len(response_text),
                              'extracted_final_answer': answer,
                              'canonical_prediction': predicted, 'canonical_target': expected,
                              'was_correct': predicted == expected,
                              'turn_status': turn_status, 'agent_message_count': len(agent_messages)}
        result['token_usage'] = token_usage
        result['tool_item_types'] = sorted({x for x in item_types if x in TOOL_TYPES})
        result['all_item_types'] = sorted({x for x in item_types if isinstance(x, str)})
        result['tool_free'] = not tool_violation
        result['settings_echo_after_turn_start'] = settings_echo
        result['request_elapsed_seconds'] = ((turn_completed_monotonic or time.monotonic()) - turn_started_monotonic)
        result['no_answer_retry'] = True
        if tool_violation:
            result['blocker'] = {'classification': 'tool_item_observed'}
    except ProtocolStop as error:
        result['status'] = 'failed_before_or_during_observation'
        result['blocker'] = {'classification': error.kind, 'method': error.method,
                             'json_rpc_error_code': error.code}
        result['token_usage'] = token_usage
        result['tool_item_types'] = sorted({x for x in item_types if x in TOOL_TYPES})
        result['settings_echo_after_turn_start'] = settings_echo
    except Exception as error:
        result['status'] = 'client_failure'
        result['blocker'] = {'classification': 'client_exception', 'type': type(error).__name__}
    finally:
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=4)
                cleanup = 'this_ephemeral_app_server_exited_after_stdio_closed'
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=8, creationflags=subprocess.CREATE_NO_WINDOW)
                    proc.wait(timeout=4)
                    cleanup = 'only_this_spawned_app_server_tree_terminated'
            except Exception as error:
                cleanup = 'cleanup_error_' + type(error).__name__
        result['helper_cleanup'] = cleanup
        result['finished_at_utc'] = utcnow()
        atomic_json(OBS / 'result.json', result)
    print(json.dumps({'status': result['status'], 'blocker': result.get('blocker'),
                      'rate_limits_before': result.get('rate_limits_before'),
                      'rate_limits_after': result.get('rate_limits_after'),
                      'catalog_evidence': result.get('catalog_evidence'),
                      'model_echo': result.get('model_echo'),
                      'settings_echo_after_turn_start': result.get('settings_echo_after_turn_start'),
                      'response': result.get('response'), 'token_usage': result.get('token_usage'),
                      'tool_item_types': result.get('tool_item_types'),
                      'request_elapsed_seconds': result.get('request_elapsed_seconds'),
                      'helper_cleanup': result.get('helper_cleanup')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
