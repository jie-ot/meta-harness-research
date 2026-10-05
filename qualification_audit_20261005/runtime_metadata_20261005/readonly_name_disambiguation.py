"""One additional authorized model/list query; no inference or account read.
Persist exactly id, model, displayName, supportedReasoningEfforts per entry.
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
FIELDS = ('id', 'model', 'displayName', 'supportedReasoningEfforts')


def main():
    marker = OUT / 'name_disambiguation_attempted.txt'
    if marker.exists():
        raise SystemExit('Authorized additional name query already attempted; do not reconnect.')
    query_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    marker.write_text(query_utc + '\n', encoding='utf-8')
    proc = None
    cleanup = 'not_started'
    catalog = None
    problem = None
    has_more = None
    try:
        deadline = time.monotonic() + 25
        proc = subprocess.Popen([NODE, CLI, 'app-server', '--stdio'], cwd=str(ROOT),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, encoding='utf-8', errors='replace', bufsize=1,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        messages = queue.Queue()

        def reader():
            try:
                for line in proc.stdout:
                    try:
                        data = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(data, dict):
                        messages.put(data)
            finally:
                messages.put(None)

        threading.Thread(target=reader, daemon=True).start()

        def rpc(identity, method, params, seconds):
            if method not in ('initialize', 'model/list'):
                raise RuntimeError('nonallowlisted_method')
            proc.stdin.write(json.dumps({'id': identity, 'method': method, 'params': params}) + '\n')
            proc.stdin.flush()
            end = min(deadline, time.monotonic() + seconds)
            while True:
                left = end - time.monotonic()
                if left <= 0:
                    raise TimeoutError(method)
                data = messages.get(timeout=left)
                if data is None:
                    raise RuntimeError('stdio_exited')
                if 'method' in data and 'id' in data:
                    raise RuntimeError('server_requires_extra_action')
                if data.get('id') != identity:
                    continue
                if 'error' in data:
                    code = data['error'].get('code') if isinstance(data['error'], dict) else None
                    raise RuntimeError('protocol_error_code_' + str(code))
                if not isinstance(data.get('result'), dict):
                    raise RuntimeError('unexpected_response_shape')
                return data['result']

        initialization = rpc(1, 'initialize', {'clientInfo': {
            'name': 'meta_harness_readonly_name_disambiguation',
            'title': 'Read-only model-name disambiguation', 'version': '1.0.0'},
            'capabilities': {'experimentalApi': False}}, 8)
        del initialization
        proc.stdin.write('{"method":"initialized"}\n')
        proc.stdin.flush()
        response = rpc(2, 'model/list', {'limit': 100, 'includeHidden': True}, 12)
        entries = response.get('data')
        if not isinstance(entries, list) or not all(isinstance(x, dict) and all(k in x for k in FIELDS) for x in entries):
            raise RuntimeError('catalog_required_fields_missing')
        catalog = [{k: x[k] for k in FIELDS} for x in entries]
        has_more = response.get('nextCursor') is not None
        (OUT / 'model_catalog_four_fields.json').write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        del response, entries
    except Exception as exc:
        # Do not log raw protocol/account/authentication responses.
        problem = type(exc).__name__
        if isinstance(exc, RuntimeError) and str(exc).startswith(('protocol_error_code_', 'catalog_required_', 'stdio_exited', 'server_requires_', 'unexpected_response_')):
            problem += ':' + str(exc)
    finally:
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=3)
                cleanup = 'this_helper_exited_after_stdio_closed'
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                    proc.wait(timeout=3)
                    cleanup = 'only_this_spawned_helper_tree_terminated'
            except Exception as exc:
                cleanup = 'cleanup_error_' + type(exc).__name__
    matches = []
    if catalog is not None:
        for x in catalog:
            name = ' '.join(x['displayName'].split()).casefold()
            if x['id'] == 'gpt-6.1-sol' or x['model'] == 'gpt-6.1-sol' or name == 'gpt-6.1 sol':
                matches.append(x)
    lines = ['只读模型名称消歧检查', '查询时间UTC：' + query_utc,
             '只发送initialize、initialized、model/list；新增推理请求0；没有登录、额度查询/变更或设置修改。',
             '本次helper清理：' + cleanup]
    if problem:
        lines.append('阻塞：' + problem)
    else:
        lines.append(f'目录返回{len(catalog)}项，includeHidden=true，后续页存在={has_more}。')
        lines.append(f'目标id/model=gpt-6.1-sol或显示名GPT-6.1 Sol的匹配项数：{len(matches)}。')
        lines.append('四类原字段保存于model_catalog_four_fields.json。')
        lines.append('不使用相近模型替代；目录不认证此前会话后端。')
    (OUT / '名称消歧结论.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'query_utc': query_utc, 'problem': problem, 'helper_cleanup': cleanup,
                      'more_pages_available': has_more, 'matching_entries': matches, 'catalog': catalog},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
