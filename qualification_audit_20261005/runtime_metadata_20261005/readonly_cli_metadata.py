"""One bounded official CLI stdio connection; read-only capability metadata.

No auth-file reads, credential extraction, inference, login or config changes.
Account responses and stderr are never logged. Only allowlisted fields persist.
"""
import datetime
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
NODE = r'D:\DevEnvs\Nodejs\node.exe'
CLI = r'D:\OpenAI\CodexBin\node_modules\@openai\codex\bin\codex.js'
SCHEMA = Path(r'C:\Users\33841\AppData\Local\Temp\codex_readonly_schema_6a80d8413ebb424aacd94f2ac42fe53f')
TARGET = 'gpt-6.1-sol'
MAX_SECONDS = 45
ALLOWED = {'initialize', 'account/read', 'account/rateLimits/read', 'model/list'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def epoch_utc(x):
    if type(x) is not int:
        return None
    try:
        return datetime.datetime.fromtimestamp(x, datetime.timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def save(data):
    p = OUT / 'metadata.json'
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class StopRead(Exception):
    def __init__(self, reason, method=None, code=None):
        self.reason, self.method, self.code = reason, method, code


def schema_check():
    x = json.loads((SCHEMA / 'ClientRequest.json').read_text(encoding='utf-8'))
    methods = {}
    for branch in x['oneOf']:
        for m in branch.get('properties', {}).get('method', {}).get('enum', []):
            if m in ALLOWED:
                methods[m] = branch
    if set(methods) != ALLOWED:
        raise StopRead('local_schema_methods_missing')
    if methods['account/rateLimits/read']['properties']['params'] != {'type': 'null'}:
        raise StopRead('local_rate_limit_parameter_schema_unexpected')
    account = json.loads((SCHEMA / 'v2/GetAccountParams.json').read_text(encoding='utf-8'))
    model = json.loads((SCHEMA / 'v2/ModelListParams.json').read_text(encoding='utf-8'))
    if account['properties']['refreshToken']['type'] != 'boolean':
        raise StopRead('local_account_parameter_schema_unexpected')
    if not {'limit', 'includeHidden'} <= set(model['properties']):
        raise StopRead('local_model_parameter_schema_unexpected')
    notification = json.loads((SCHEMA / 'ClientNotification.json').read_text(encoding='utf-8'))
    if not any(b.get('required') == ['method'] and b.get('properties', {}).get('method', {}).get('enum') == ['initialized'] for b in notification['oneOf']):
        raise StopRead('local_initialized_schema_unexpected')


def error_kind(error):
    # Examine only for classification; never persist the server's raw message.
    code = error.get('code') if isinstance(error, dict) else None
    message = str(error.get('message', '')).lower() if isinstance(error, dict) else ''
    if code == -32601:
        return 'method_unsupported'
    if any(t in message for t in ('login', 'sign in', 'unauthorized', 'authentication', '401', '403', 'authorize', 'authorization')):
        return 'authentication_or_authorization_required'
    if 'gateway' in message:
        return 'gateway_unavailable_or_authorization_required'
    if code == -32602:
        return 'invalid_params'
    return 'read_request_failed'


def window(x):
    if not isinstance(x, dict):
        return None
    used, minutes = x.get('usedPercent'), x.get('windowDurationMins')
    return {'window_duration_minutes': minutes,
            'window_kind': 'weekly_7_days' if minutes == 10080 else 'other_or_unknown',
            'used_percent': used,
            'remaining_percent_derived': 100 - used if type(used) in (int, float) and 0 <= used <= 100 else None,
            'resets_at_epoch_seconds': x.get('resetsAt'),
            'resets_at_utc': epoch_utc(x.get('resetsAt'))}


def limit_snapshot(x, key, default_view=False):
    return {'bucket_map_key': key, 'limit_id': x.get('limitId'), 'limit_name': x.get('limitName'),
            'is_backward_compatible_default_view': default_view,
            'primary': window(x.get('primary')), 'secondary': window(x.get('secondary')),
            'target_model_bucket_link': 'unknown: model/list schema has no rate-limit bucket mapping'}


def main():
    if (OUT / 'connection_started.json').exists():
        raise SystemExit('One bounded connection already attempted; no automatic reconnect.')
    result = {'cli_version': '0.149.1', 'cli_package': '@openai/codex', 'started_at_utc': now(),
              'transport': 'stdio', 'max_connection_seconds': MAX_SECONDS,
              'status': 'starting', 'account': None, 'rate_limits': None, 'target_model': None,
              'gateway_readiness': 'not requested: no such standard method in generated local schema',
              'operations_sent': [], 'protocol_corrections': 0,
              'inference_operations_sent': 0,
              'catalog_does_not_attest_previous_session_backend': True,
              'percent_hardcap_control_established': False,
              'persistent_settings_changed': False,
              'authentication_files_or_tokens_manually_read': False}
    proc = None
    cleanup = 'not_started'
    try:
        schema_check()
        (OUT / 'connection_started.json').write_text(json.dumps({'time_utc': result['started_at_utc'],
            'clientInfo': {'name': 'meta_harness_readonly_capability_audit',
                           'title': 'Read-only local capability audit', 'version': '1.0.0'},
            'allowed_methods': sorted(ALLOWED), 'max_seconds': MAX_SECONDS}, indent=2) + '\n', encoding='utf-8')
        start = time.monotonic()
        deadline = start + MAX_SECONDS
        proc = subprocess.Popen([NODE, CLI, 'app-server', '--stdio'], cwd=str(ROOT),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, encoding='utf-8', errors='replace', bufsize=1,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        messages = queue.Queue()

        def reader():
            try:
                for line in proc.stdout:
                    try:
                        x = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(x, dict):
                        messages.put(x)
            finally:
                messages.put(None)

        threading.Thread(target=reader, daemon=True).start()
        next_id = 0

        def rpc(method, params=None, include_params=True, timeout=12):
            nonlocal next_id
            if method not in ALLOWED:
                raise StopRead('method_outside_allowlist', method)
            next_id += 1
            request = {'id': next_id, 'method': method}
            if include_params:
                request['params'] = params
            result['operations_sent'].append({'method': method, 'sent_at_utc': now()})
            if method == 'account/read':
                result['operations_sent'][-1]['refreshToken'] = False
            save(result)
            proc.stdin.write(json.dumps(request) + '\n')
            proc.stdin.flush()
            until = min(deadline, time.monotonic() + timeout)
            while True:
                left = until - time.monotonic()
                if left <= 0:
                    raise StopRead('timeout', method)
                try:
                    response = messages.get(timeout=left)
                except queue.Empty:
                    raise StopRead('timeout', method)
                if response is None:
                    raise StopRead('stdio_process_exited', method)
                if 'method' in response and 'id' in response:
                    raise StopRead('server_requires_nonallowlisted_client_action', method)
                if response.get('id') != next_id:
                    # Notifications are discarded, never persisted.
                    continue
                if 'error' in response:
                    e = response['error']
                    raise StopRead(error_kind(e), method, e.get('code') if isinstance(e, dict) else None)
                if not isinstance(response.get('result'), dict):
                    raise StopRead('unexpected_result_shape', method)
                return response['result']

        initialization = rpc('initialize', {'clientInfo': {'name': 'meta_harness_readonly_capability_audit',
                        'title': 'Read-only local capability audit', 'version': '1.0.0'},
                        'capabilities': {'experimentalApi': False}}, timeout=10)
        result['initialize_succeeded'] = True
        result['initialize_version_matches_help'] = '0.149.1' in str(initialization.get('userAgent', ''))
        del initialization
        proc.stdin.write('{"method":"initialized"}\n')
        proc.stdin.flush()
        result['operations_sent'].append({'method': 'initialized', 'sent_at_utc': now()})
        account = rpc('account/read', {'refreshToken': False}, timeout=8)
        a = account.get('account')
        kind = a.get('type') if isinstance(a, dict) else None
        result['account'] = {'account_type': kind, 'requires_openai_auth': account.get('requiresOpenaiAuth'),
                             'chatgpt_subscription_login_present': kind == 'chatgpt',
                             'subscription_rate_limit_query_succeeded': False}
        del account, a
        save(result)
        if kind is None:
            raise StopRead('existing_login_unavailable', 'account/read')
        if kind != 'chatgpt':
            raise StopRead('existing_login_not_chatgpt_subscription', 'account/read')
        quota_time = now()
        rates = rpc('account/rateLimits/read', include_params=False, timeout=12)
        buckets = []
        by_id = rates.get('rateLimitsByLimitId')
        if isinstance(by_id, dict):
            for key, x in by_id.items():
                if isinstance(x, dict):
                    buckets.append(limit_snapshot(x, key))
        legacy = rates.get('rateLimits')
        if isinstance(legacy, dict):
            same = [i for i, x in enumerate(buckets)
                    if x['limit_id'] == legacy.get('limitId') and x['primary'] == window(legacy.get('primary'))
                    and x['secondary'] == window(legacy.get('secondary'))]
            if same:
                buckets[same[0]]['is_backward_compatible_default_view'] = True
            else:
                buckets.append(limit_snapshot(legacy, None, True))
        result['rate_limits'] = {'queried_at_utc': quota_time, 'response_received_at_utc': now(),
                                 'buckets': buckets,
                                 'model_bucket_mapping': 'unknown; catalog has no mapping field; do not assume every bucket applies to gpt-6.1-sol',
                                 'remaining_calculation': '100-usedPercent, for each reported window separately'}
        result['account']['subscription_rate_limit_query_succeeded'] = True
        del rates, by_id, legacy
        save(result)
        model_time = now()
        models = rpc('model/list', {'limit': 100, 'includeHidden': True}, timeout=12)
        data = models.get('data')
        if not isinstance(data, list):
            raise StopRead('unexpected_model_catalog_shape', 'model/list')
        target = [x for x in data if isinstance(x, dict) and (x.get('id') == TARGET or x.get('model') == TARGET)]
        matched = []
        for x in target:
            efforts = [p.get('reasoningEffort') for p in x.get('supportedReasoningEfforts', []) if isinstance(p, dict)]
            matched.append({'id': x.get('id'), 'model': x.get('model'), 'display_name': x.get('displayName'),
                            'hidden': x.get('hidden'), 'is_default': x.get('isDefault'),
                            'supported_reasoning_efforts': efforts,
                            'literal_max_supported': any(isinstance(e, str) and e.lower() == 'max' for e in efforts),
                            'default_reasoning_effort': x.get('defaultReasoningEffort')})
        result['target_model'] = {'requested': TARGET, 'queried_at_utc': model_time,
                                  'catalog_items_returned': len(data), 'more_pages_available': models.get('nextCursor') is not None,
                                  'listed_in_returned_catalog': bool(matched), 'matching_entries': matched,
                                  'availability_limit': 'catalog only; no request executed or previous backend authenticated'}
        del models, data, target
        result['status'] = 'read_only_queries_succeeded'
    except StopRead as e:
        result['status'] = 'stopped'
        result['blocker'] = {'classification': e.reason, 'method': e.method, 'json_rpc_error_code': e.code}
    except Exception as e:
        result['status'] = 'stopped'
        result['blocker'] = {'classification': 'client_exception', 'exception_type': type(e).__name__}
    finally:
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=3)
                cleanup = 'this_helper_exited_after_stdio_closed'
            except subprocess.TimeoutExpired:
                # PID belongs to the one process spawned above. Never enumerate
                # or stop other Codex processes. /T covers this helper's child.
                if proc.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                    proc.wait(timeout=3)
                    cleanup = 'only_this_spawned_helper_tree_terminated'
            except Exception as e:
                cleanup = 'cleanup_error_' + type(e).__name__
        result['helper_cleanup'] = cleanup
        result['finished_at_utc'] = now()
        save(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
