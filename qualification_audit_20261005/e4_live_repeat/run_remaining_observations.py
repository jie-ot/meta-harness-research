"""Run at most the 39 authorized remaining E4 observations, strictly serial.

Every observation gets a new ephemeral thread. Before every turn the script
checks the 7-day Codex bucket and stops when displayed remaining <= 2%.
There are no request or answer retries, model substitutions, or project imports.
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
ROOT = OUT.parents[1]
BATCH = OUT / 'batch_observations'
NODE = r'D:\DevEnvs\Nodejs\node.exe'
CLI = r'D:\OpenAI\CodexBin\node_modules\@openai\codex\bin\codex.js'
MODEL = 'gpt-5.6-sol'
EFFORT = 'max'
POLICY_HASH = 'b8b08cec89be58d517539dea639b600290e6181eef84b99a9c4f83ac1d2845f3'
REQUEST_CONFIG_HASH = '3dacf214af4260a058ff025f5d7b096e9a9a6b4b3abd3e9f864c006a8b55614b'
STOP_REMAINING = 2
MAX_TURNS = 39
PROCESS_TIMEOUT_SECONDS = 1800
TURN_TIMEOUT_SECONDS = 300
DISABLED_FEATURES = ['shell_tool', 'unified_exec', 'shell_snapshot', 'memories',
                     'apps', 'browser_use', 'computer_use', 'image_generation',
                     'multi_agent', 'plugins', 'view_image']
TOOL_TYPES = {'commandExecution', 'fileChange', 'mcpToolCall', 'dynamicToolCall',
              'collabAgentToolCall', 'webSearch', 'imageView', 'imageGeneration'}


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text_digest(text):
    return digest(text.encode('utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


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


def weekly_snapshot(response, queried_at, received_at):
    by_id = response.get('rateLimitsByLimitId')
    value = by_id.get('codex') if isinstance(by_id, dict) and isinstance(by_id.get('codex'), dict) else response.get('rateLimits')
    if not isinstance(value, dict):
        return {'valid': False, 'queried_at_utc': queried_at, 'received_at_utc': received_at}
    primary = value.get('primary')
    if not isinstance(primary, dict):
        return {'valid': False, 'queried_at_utc': queried_at, 'received_at_utc': received_at}
    used = primary.get('usedPercent')
    duration = primary.get('windowDurationMins')
    return {'valid': type(used) is int and duration == 10080,
            'queried_at_utc': queried_at, 'received_at_utc': received_at,
            'bucket_map_key': 'codex' if isinstance(by_id, dict) and 'codex' in by_id else None,
            'limit_id': value.get('limitId'), 'window_duration_minutes': duration,
            'used_percent': used,
            'remaining_percent_derived': 100 - used if type(used) is int and 0 <= used <= 100 else None,
            'resets_at_epoch_seconds': primary.get('resetsAt')}


class StopBatch(Exception):
    def __init__(self, kind, method=None, code=None):
        self.kind, self.method, self.code = kind, method, code


def main():
    marker = OUT / 'batch_started.json'
    if marker.exists():
        raise SystemExit('Batch was already started. Automatic restart is forbidden.')
    policy_bytes = (OUT / 'batch_policy.json').read_bytes()
    config_bytes = (OUT / 'request_config.json').read_bytes()
    if digest(policy_bytes) != POLICY_HASH or digest(config_bytes) != REQUEST_CONFIG_HASH:
        raise SystemExit('Frozen batch policy or request config changed.')
    policy = json.loads(policy_bytes)
    config = json.loads(config_bytes)
    locked = [json.loads(x) for x in (OUT / 'locked_items.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    first = json.loads((OUT / 'observation_000_live/result.json').read_text(encoding='utf-8'))
    wrapper = json.loads((OUT / 'instruction_source_assessment.json').read_text(encoding='utf-8'))
    feature_check = json.loads((OUT / 'tool_feature_precheck.json').read_text(encoding='utf-8'))
    schedule = [(entry['pass'], index) for entry in policy['schedule'] for index in entry['locked_item_indices']]
    if (len(locked) != 20 or len(schedule) != MAX_TURNS or schedule[0] != (1, 1)
            or schedule[-1] != (2, 19) or first['status'] != 'success'
            or first['turn_start_requests_sent'] != 1
            or wrapper.get('accept_as_fixed_wrapper') is not True
            or not all(v is False for v in feature_check['effective_states'].values())):
        raise SystemExit('Frozen schedule/prior result/wrapper/tool prerequisites failed.')
    source_expected = wrapper['sources'][0]
    if source_expected['task_data_detected'] is not False:
        raise SystemExit('Instruction source task-data audit is not clean.')
    for item in locked:
        prompt_path = OUT / item['prompt_path']
        if digest(prompt_path.read_bytes()) != item['prompt_sha256']:
            raise SystemExit('A locked prompt byte hash changed.')
    BATCH.mkdir(exist_ok=True)
    state = {'status': 'starting', 'started_at_utc': utcnow(), 'policy_sha256': POLICY_HASH,
             'model': MODEL, 'effort': EFFORT, 'quota_stop_remaining_lte': STOP_REMAINING,
             'maximum_new_turns': MAX_TURNS, 'turn_start_requests_sent': 0,
             'completed_observations': 0, 'failed_observations': 0,
             'next_schedule_position': 0, 'stop_reason': None, 'observations': [],
             'remaining_39_not_mandatory': True, 'answer_retries': 0,
             'model_substitutions': 0, 'credit_resets_or_paid_api': 0}
    write_json(marker, {'started_at_utc': state['started_at_utc'], 'policy_sha256': POLICY_HASH,
                        'schedule': schedule, 'actual_turn_limit': MAX_TURNS,
                        'stop_condition': policy['weekly_quota_stop_condition']})
    write_json(OUT / 'batch_state.json', state)
    proc = None
    cleanup = 'not_started'
    messages = queue.Queue()
    request_id = 0
    process_deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
    current = None

    def persist():
        write_json(OUT / 'batch_state.json', state)

    try:
        command = [NODE, CLI, 'app-server', '--stdio']
        for feature in DISABLED_FEATURES:
            command.extend(['--disable', feature])
        proc = subprocess.Popen(command, cwd=str(OUT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace',
                                bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW)

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

        def deny_server_request(value):
            proc.stdin.write(json.dumps({'id': value.get('id'),
                                         'error': {'code': -32000, 'message': 'E4 adapter rejects all tool/client actions'}}) + '\n')
            proc.stdin.flush()

        def handle(value):
            nonlocal current
            method = value.get('method')
            if not isinstance(method, str) or current is None:
                return
            params = value.get('params') if isinstance(value.get('params'), dict) else {}
            event = {'method': method, 'received_at_utc': utcnow()}
            if method in ('item/started', 'item/completed'):
                item_value = params.get('item') if isinstance(params.get('item'), dict) else {}
                item_type = item_value.get('type')
                event['item_type'] = item_type
                current['item_types'].append(item_type)
                if item_type in TOOL_TYPES:
                    current['tool_violation'] = True
                    event['protocol_violation'] = True
                    if current.get('thread_id') and current.get('turn_id'):
                        send('turn/interrupt', {'threadId': current['thread_id'], 'turnId': current['turn_id']})
                if method == 'item/completed' and item_type == 'agentMessage':
                    current['agent_messages'].append({'id': item_value.get('id'),
                                                       'phase': item_value.get('phase'),
                                                       'text': item_value.get('text', '')})
            elif method == 'thread/tokenUsage/updated':
                current['token_usage'] = params.get('tokenUsage')
                event['usage_present'] = isinstance(current['token_usage'], dict)
            elif method == 'thread/settings/updated':
                settings = params.get('threadSettings') if isinstance(params.get('threadSettings'), dict) else {}
                current['settings_echo'] = {k: settings.get(k) for k in ('model', 'modelProvider', 'effort',
                                                                          'summary', 'approvalPolicy',
                                                                          'sandboxPolicy', 'cwd')}
                event['settings_echo_recorded'] = True
            elif method == 'turn/started':
                turn = params.get('turn') if isinstance(params.get('turn'), dict) else {}
                current['turn_id'] = turn.get('id') or current.get('turn_id')
                event['turn_id'] = current['turn_id']
            elif method == 'turn/completed':
                current['turn_complete'] = params.get('turn') if isinstance(params.get('turn'), dict) else {}
                current['turn_id'] = current['turn_complete'].get('id') or current.get('turn_id')
                current['turn_completed_monotonic'] = time.monotonic()
                event['turn_status'] = current['turn_complete'].get('status')
            current['events'].append(event)
            write_json(current['directory'] / 'event_summary.json', current['events'])

        def rpc(method, params=None, include_params=True, timeout=20):
            identity = send(method, params, include_params)
            end = min(process_deadline, time.monotonic() + timeout)
            while True:
                left = end - time.monotonic()
                if left <= 0:
                    raise StopBatch('timeout', method)
                try:
                    value = messages.get(timeout=left)
                except queue.Empty:
                    raise StopBatch('timeout', method)
                if value is None:
                    raise StopBatch('stdio_process_exited', method)
                if 'method' in value and 'id' in value:
                    deny_server_request(value)
                    if current is not None:
                        current['client_action_request_rejected'] = True
                    raise StopBatch('server_requested_client_action', method)
                if 'method' in value:
                    handle(value)
                    continue
                if value.get('id') != identity:
                    continue
                if 'error' in value:
                    error = value['error'] if isinstance(value['error'], dict) else {}
                    raise StopBatch('json_rpc_error', method, error.get('code'))
                if not isinstance(value.get('result'), dict):
                    raise StopBatch('unexpected_result_shape', method)
                return value['result']

        initialized = rpc('initialize', {'clientInfo': {'name': 'meta_harness_e4_serial_batch',
                          'title': 'Meta-Harness E4 serial repeat', 'version': '1.0.0'},
                          'capabilities': {'experimentalApi': False}}, timeout=10)
        state['cli_user_agent'] = initialized.get('userAgent')
        proc.stdin.write('{"method":"initialized"}\n')
        proc.stdin.flush()
        catalog = rpc('model/list', {'limit': 100, 'includeHidden': True}, timeout=15)
        matches = [x for x in catalog.get('data', []) if isinstance(x, dict)
                   and (x.get('id') == MODEL or x.get('model') == MODEL)]
        efforts = [x.get('reasoningEffort') for x in matches[0].get('supportedReasoningEfforts', [])
                   if isinstance(x, dict)] if len(matches) == 1 else []
        if len(matches) != 1 or EFFORT not in efforts or catalog.get('nextCursor') is not None:
            raise StopBatch('model_or_effort_unavailable', 'model/list')
        state['catalog_evidence'] = {'id': matches[0].get('id'), 'model': matches[0].get('model'),
                                     'display_name': matches[0].get('displayName'),
                                     'supported_reasoning_efforts': efforts, 'catalog_complete': True}
        state['status'] = 'running'
        persist()
        for position, (pass_number, item_index) in enumerate(schedule):
            item = locked[item_index]
            observation_id = f'pass{pass_number}_item{item_index:02d}'
            directory = BATCH / observation_id
            directory.mkdir(exist_ok=False)
            prompt = (OUT / item['prompt_path']).read_bytes().decode('utf-8')
            current = {'observation_id': observation_id, 'pass': pass_number,
                       'locked_item_index': item_index, 'item_id': item['item_id'],
                       'directory': directory, 'events': [], 'agent_messages': [],
                       'item_types': [], 'token_usage': None, 'settings_echo': None,
                       'tool_violation': False, 'client_action_request_rejected': False,
                       'thread_id': None, 'turn_id': None, 'turn_complete': None,
                       'turn_start_sent': False, 'started_at_utc': utcnow()}
            write_json(directory / 'attempt.json', {k: v for k, v in current.items()
                                                    if k not in ('directory', 'events', 'agent_messages')})
            if text_digest(prompt) != item['prompt_sha256']:
                raise StopBatch('prompt_hash_mismatch_before_request', observation_id)
            query_time = utcnow()
            rate_before = weekly_snapshot(rpc('account/rateLimits/read', include_params=False, timeout=15),
                                          query_time, utcnow())
            current['rate_before'] = rate_before
            write_json(directory / 'progress.json', {k: v for k, v in current.items() if k != 'directory'})
            if not rate_before['valid']:
                raise StopBatch('weekly_rate_limit_snapshot_invalid', observation_id)
            if rate_before['remaining_percent_derived'] <= STOP_REMAINING:
                state['stop_reason'] = {'kind': 'weekly_safety_margin_reached',
                                        'before_observation': observation_id,
                                        'remaining_percent': rate_before['remaining_percent_derived']}
                state['status'] = 'stopped_by_quota_guard'
                state['next_schedule_position'] = position
                persist()
                write_json(directory / 'not_sent.json', state['stop_reason'])
                break
            thread = rpc('thread/start', config['thread_start'], timeout=20)
            thread_value = thread.get('thread') if isinstance(thread.get('thread'), dict) else {}
            current['thread_id'] = thread_value.get('id')
            sources = thread.get('instructionSources') if isinstance(thread.get('instructionSources'), list) else []
            source_rows = []
            for source in sources:
                if not isinstance(source, str):
                    source_rows.append({'valid': False})
                    continue
                path = Path(source)
                normalized = str(path.resolve()) if path.is_absolute() else source
                source_rows.append({'valid': path.is_file(), 'path_sha256': digest(normalized.encode('utf-8')),
                                    'content_sha256': digest(path.read_bytes()) if path.is_file() else None,
                                    'bytes': path.stat().st_size if path.is_file() else None})
            source_match = (len(source_rows) == 1 and source_rows[0].get('valid') is True
                            and source_rows[0].get('path_sha256') == source_expected['path_sha256']
                            and source_rows[0].get('content_sha256') == source_expected['content_sha256']
                            and source_rows[0].get('bytes') == source_expected['bytes'])
            current['thread_echo'] = {'model': thread.get('model'), 'model_provider': thread.get('modelProvider'),
                                      'reasoning_effort': thread.get('reasoningEffort'),
                                      'ephemeral': thread_value.get('ephemeral'),
                                      'approval_policy': thread.get('approvalPolicy'),
                                      'sandbox': thread.get('sandbox'),
                                      'instruction_source_count': len(sources),
                                      'wrapper_hash_match': source_match}
            write_json(directory / 'progress.json', {k: v for k, v in current.items() if k != 'directory'})
            if (not current['thread_id'] or thread.get('model') != MODEL
                    or thread.get('reasoningEffort') != EFFORT
                    or thread_value.get('ephemeral') is not True or not source_match):
                raise StopBatch('thread_protocol_mismatch', observation_id)
            turn_params = dict(config['turn_start'])
            turn_params.pop('input_rule', None)
            turn_params.pop('output_schema', None)
            turn_params['threadId'] = current['thread_id']
            turn_params['input'] = [{'type': 'text', 'text': prompt}]
            if len(turn_params['input']) != 1 or text_digest(turn_params['input'][0]['text']) != item['prompt_sha256']:
                raise StopBatch('outgoing_prompt_mismatch', observation_id)
            current['turn_start_sent'] = True
            current['turn_started_monotonic'] = time.monotonic()
            state['turn_start_requests_sent'] += 1
            state['next_schedule_position'] = position
            persist()
            write_json(directory / 'progress.json', {k: v for k, v in current.items() if k != 'directory'})
            response = rpc('turn/start', turn_params, timeout=25)
            turn_value = response.get('turn') if isinstance(response.get('turn'), dict) else {}
            current['turn_id'] = turn_value.get('id') or current['turn_id']
            end = min(process_deadline, current['turn_started_monotonic'] + TURN_TIMEOUT_SECONDS)
            while current['turn_complete'] is None:
                left = end - time.monotonic()
                if left <= 0:
                    raise StopBatch('turn_completion_timeout', observation_id)
                try:
                    value = messages.get(timeout=min(left, 15))
                except queue.Empty:
                    continue
                if value is None:
                    raise StopBatch('stdio_process_exited_during_turn', observation_id)
                if 'method' in value and 'id' in value:
                    deny_server_request(value)
                    current['client_action_request_rejected'] = True
                    raise StopBatch('server_requested_client_action', observation_id)
                if 'method' in value:
                    handle(value)
            after_query = utcnow()
            rate_after = weekly_snapshot(rpc('account/rateLimits/read', include_params=False, timeout=15),
                                         after_query, utcnow())
            current['rate_after'] = rate_after
            final_messages = [x for x in current['agent_messages'] if x.get('phase') == 'final'] or current['agent_messages']
            response_text = final_messages[-1]['text'] if final_messages else ''
            (directory / 'response.txt').write_bytes(response_text.encode('utf-8'))
            write_json(directory / 'agent_messages.json', current['agent_messages'])
            answer = extract_json_field(response_text, 'final_answer')
            completed_text = current['turn_complete'].get('status') == 'completed' and bool(response_text)
            predicted = canonical(answer) if completed_text else None
            target = canonical(item['target']) if completed_text else None
            result = {'observation_id': observation_id, 'pass': pass_number,
                      'locked_item_index': item_index, 'item_id': item['item_id'],
                      'status': 'completed' if completed_text and not current['tool_violation'] else 'failed',
                      'failure_reason': None if completed_text and not current['tool_violation'] else
                          ('tool_item_observed' if current['tool_violation'] else 'no_completed_final_text'),
                      'model': MODEL, 'effort': EFFORT, 'thread_id': current['thread_id'],
                      'turn_id': current['turn_id'], 'thread_echo': current['thread_echo'],
                      'prompt_sha256': item['prompt_sha256'], 'prompt_chars': item['prompt_chars'],
                      'response_sha256': text_digest(response_text), 'response_chars': len(response_text),
                      'extracted_final_answer': answer if completed_text else None,
                      'format_valid_final_answer': bool(answer) if completed_text else None,
                      'canonical_prediction': predicted, 'canonical_target': target,
                      'was_correct': predicted == target if completed_text else None,
                      'elapsed_seconds': (current.get('turn_completed_monotonic', time.monotonic())
                                          - current['turn_started_monotonic']),
                      'token_usage': current['token_usage'], 'rate_before': rate_before,
                      'rate_after': rate_after, 'tool_item_types': sorted({x for x in current['item_types'] if x in TOOL_TYPES}),
                      'all_item_types': sorted({x for x in current['item_types'] if isinstance(x, str)}),
                      'client_action_request_rejected': current['client_action_request_rejected'],
                      'answer_retries': 0, 'request_retries': 0,
                      'cached_input_tokens': (current['token_usage'] or {}).get('last', {}).get('cachedInputTokens'),
                      'completed_at_utc': utcnow()}
            write_json(directory / 'result.json', result)
            if result['status'] == 'completed':
                state['completed_observations'] += 1
            else:
                state['failed_observations'] += 1
            state['observations'].append({'observation_id': observation_id, 'pass': pass_number,
                                          'locked_item_index': item_index, 'item_id': item['item_id'],
                                          'status': result['status'], 'was_correct': result['was_correct'],
                                          'remaining_before': rate_before['remaining_percent_derived'],
                                          'remaining_after': rate_after.get('remaining_percent_derived'),
                                          'total_tokens': (current['token_usage'] or {}).get('last', {}).get('totalTokens'),
                                          'elapsed_seconds': result['elapsed_seconds']})
            state['next_schedule_position'] = position + 1
            persist()
            print(json.dumps({'progress': f'{position + 1}/{MAX_TURNS}', 'observation_id': observation_id,
                              'item_id': item['item_id'], 'status': result['status'],
                              'correct': result['was_correct'],
                              'tokens': state['observations'][-1]['total_tokens'],
                              'elapsed_seconds': round(result['elapsed_seconds'], 3),
                              'remaining_before': rate_before['remaining_percent_derived'],
                              'remaining_after': rate_after.get('remaining_percent_derived')},
                             ensure_ascii=False), flush=True)
            if result['status'] != 'completed' or current['token_usage'] is None or not rate_after['valid']:
                raise StopBatch('observation_or_logging_incomplete', observation_id)
            current = None
        else:
            state['status'] = 'completed_all_authorized_observations'
            state['stop_reason'] = {'kind': 'schedule_complete'}
            persist()
    except StopBatch as error:
        if current is not None:
            failure = {'status': 'failed', 'observation_id': current['observation_id'],
                       'pass': current['pass'], 'locked_item_index': current['locked_item_index'],
                       'item_id': current['item_id'], 'turn_start_sent': current['turn_start_sent'],
                       'failure_reason': error.kind, 'failure_method_or_observation': error.method,
                       'json_rpc_error_code': error.code, 'token_usage': current['token_usage'],
                       'rate_before': current.get('rate_before'), 'tool_violation': current['tool_violation'],
                       'client_action_request_rejected': current['client_action_request_rejected'],
                       'events_saved': len(current['events']), 'answer_retries': 0,
                       'request_retries': 0, 'failed_at_utc': utcnow()}
            write_json(current['directory'] / 'failure.json', failure)
        state['status'] = 'stopped_on_failure'
        state['stop_reason'] = {'kind': error.kind, 'method_or_observation': error.method,
                                'json_rpc_error_code': error.code}
        persist()
    except Exception as error:
        if current is not None:
            write_json(current['directory'] / 'failure.json',
                       {'status': 'failed', 'observation_id': current['observation_id'],
                        'turn_start_sent': current['turn_start_sent'],
                        'failure_reason': 'client_exception', 'exception_type': type(error).__name__,
                        'events_saved': len(current['events']), 'answer_retries': 0,
                        'request_retries': 0, 'failed_at_utc': utcnow()})
        state['status'] = 'stopped_on_failure'
        state['stop_reason'] = {'kind': 'client_exception', 'exception_type': type(error).__name__}
        persist()
    finally:
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=5)
                cleanup = 'this_batch_app_server_exited_after_stdio_closed'
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=8, creationflags=subprocess.CREATE_NO_WINDOW)
                    proc.wait(timeout=5)
                    cleanup = 'only_this_batch_app_server_tree_terminated'
            except Exception as error:
                cleanup = 'cleanup_error_' + type(error).__name__
        state['helper_cleanup'] = cleanup
        state['finished_at_utc'] = utcnow()
        persist()
    print(json.dumps({'batch_final_status': state['status'],
                      'turn_start_requests_sent': state['turn_start_requests_sent'],
                      'completed_observations': state['completed_observations'],
                      'failed_observations': state['failed_observations'],
                      'next_schedule_position': state['next_schedule_position'],
                      'stop_reason': state['stop_reason'], 'helper_cleanup': state['helper_cleanup']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
