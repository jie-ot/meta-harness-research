"""Finalize E4 preflight after the strict isolation gate stopped pre-inference."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OBS = OUT / 'observation_000'
EXPECTED = {
    'agent': ('lawbench_rsi_v3/engine/text_classification/agents/confusion_disambiguation_memory.py',
              '29cfaead2ebda91a803daee6f2460e53707fc5b804fe3d2135859d350cd4ee8b'),
    'memory': ('reference_examples/text_classification/logs/20260925_200856/LawBench/confusion_disambiguation_memory/gpt-oss-120b/memory.json',
               '841af6631f324cf3297eb12826316ee73b484c329b6484ddba8df3c94c805f79'),
    'manifest': ('lawbench_rsi_v3/external/prepared_data_v3/manifest_v3.json',
                 '486493d2b3b6241ff92aeca5d18e634b77fd48412ce6fdad1452406f65e1532d'),
    'score': ('lawbench_rsi_v3/external/prepared_data_v3/score.jsonl',
              '0732d37bbfaa29ef8b07d698a7d48c4b8623817ac6bd4ab3bd61af37b173c417'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


result = json.loads((OBS / 'result.json').read_text(encoding='utf-8'))
lock = json.loads((OUT / 'lock_manifest.json').read_text(encoding='utf-8'))
request = json.loads((OUT / 'request_config.json').read_text(encoding='utf-8'))
selfcheck = json.loads((OUT / 'pre_request_selfcheck.json').read_text(encoding='utf-8'))
source_checks = {}
for name, (relative, expected) in EXPECTED.items():
    actual = sha(ROOT / relative)
    source_checks[name] = {'path': relative, 'expected_sha256': expected,
                           'actual_sha256': actual, 'unchanged': actual == expected}
before = result['rate_limits_before']
codex = next((x for x in before['buckets'] if x['bucket_map_key'] == 'codex'), None) if before else None
primary = codex.get('primary') if codex else None
report = {
    'status': 'STOPPED_BEFORE_CLASSIFICATION_INFERENCE',
    'reason': 'thread/start reported one additional instruction source; strict original-harness-only semantic isolation was not confirmed',
    'formal_observation': {'item_id': lock['selected_item_ids'][0],
                           'turn_start_requests_sent': result['turn_start_requests_sent'],
                           'classification_prompt_sent': False,
                           'answer': None, 'correctness': None,
                           'request_elapsed_seconds': None,
                           'token_usage': None, 'answer_retries': 0,
                           'tool_calls': 0},
    'locked_design': {'selection': lock['selection_rule'], 'items': len(lock['selected_item_ids']),
                      'selected_item_ids_sha256': lock['selected_item_ids_sha256'],
                      'first_prompt_sha256': result['attempt']['prompt_sha256'],
                      'first_prompt_chars': result['attempt']['prompt_chars'],
                      'historical_exact_prompt_matches': lock['prompt_validation']['items_exactly_matching_input_target_actual_and_constructed_prompt'],
                      'per_observation_reset': request['per_observation_reset']},
    'frozen_hashes': lock['frozen_hashes'],
    'source_recheck': source_checks,
    'model_capability_evidence': result['catalog_evidence'],
    'thread_start_echo': result['model_echo'],
    'semantic_differences': [
        'Codex app-server is an agent transport; raw completion equivalence cannot be proven.',
        'CLI 0.149.1 exposes no turn/start temperature or max-output-token field.',
        'Core tool declarations have no disable field; execution would have been invalidated if a tool item appeared.',
        'One instruction source was loaded despite an isolated cwd and blank developerInstructions; its content/path was not retained.',
    ],
    'quota': {'before': before, 'after': None,
              'after_missing_reason': 'strict isolation gate stopped before turn/start; no inference/token event occurred',
              'latest_weekly_used_percent': primary.get('used_percent') if primary else None,
              'latest_weekly_remaining_percent': primary.get('remaining_percent_derived') if primary else None},
    'full_40_budget_judgment': {
        'verdict': 'NOT_ESTABLISHED_AND_NOT_CONSERVATIVELY_APPROVED',
        'basis': 'No task inference occurred, so there is no empirical token or quota delta. The latest coarse weekly snapshot shows 12% remaining, and bucket-to-model accounting is unspecified.',
        'remaining_requests_started': 0,
    },
    'implementation_files': ['prepare_e4.py', 'run_first_observation.py',
                             'repair_prompt_bytes_before_live.py', 'finalize_preflight.py'],
    'evidence_files': ['lock_manifest.json', 'locked_items.jsonl', 'request_config.json',
                       'pre_request_selfcheck.json', 'prompt_byte_repair.json',
                       'observation_000/result.json', 'observation_000/event_summary.json'],
    'project_instruction_discovery': {'project_AGENTS_md_found': False,
                                      'project_or_parent_dot_agents_found': False,
                                      'note': 'bounded checks covered meta-harness and its immediate research-work parent'},
    'safety': {'new_model_inference_turns': 0, 'remaining_39_started': False,
               'source_files_modified': False, 'credentials_read_or_copied': False,
               'login_or_permissions_changed': False, 'software_installed': False,
               'helper_cleanup': result['helper_cleanup']},
}
save(OUT / 'final_report.json', report)
checks = {
    'all_pre_request_checks_passed': selfcheck['status'] == 'PASS',
    'twenty_inputs_locked': len(lock['selected_item_ids']) == 20,
    'all_twenty_prompts_historical_exact': lock['prompt_validation']['items_exactly_matching_input_target_actual_and_constructed_prompt'] == 20,
    'exact_model_catalogued_with_max': result['catalog_evidence']['id'] == 'gpt-5.6-sol' and result['catalog_evidence']['max_supported'],
    'exact_model_and_effort_echoed': result['model_echo']['model'] == 'gpt-5.6-sol' and result['model_echo']['reasoning_effort'] == 'max',
    'strict_gate_detected_instruction_source': result['model_echo']['instruction_source_count'] == 1,
    'no_classification_turn_sent': result['turn_start_requests_sent'] == 0,
    'no_token_usage': result['token_usage'] is None,
    'no_tool_items': not result['tool_item_types'],
    'all_frozen_sources_unchanged': all(x['unchanged'] for x in source_checks.values()),
    'ephemeral_helper_exited': result['helper_cleanup'] == 'this_ephemeral_app_server_exited_after_stdio_closed',
}
review = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
          'meaning': 'implementation preflight correctly stopped before inference; not an experimental success'}
save(OUT / 'output_review.json', review)
lines = [
    'E4实施前检查与最小链路验证：在首条分类推理前停止',
    '',
    'R4B代码、trained memory、manifest和score split仍匹配冻结哈希。manifest前20项已在结果前锁定；独立渲染的20个runtime input、target、constructed/actual prompt均与历史R4B逐字一致。',
    '首条为lawbench_3-3_0481，prompt 30,802字符，SHA-256=' + result['attempt']['prompt_sha256'] + '。',
    '',
    '官方CLI目录确认gpt-5.6-sol支持max；thread/start也回显model=gpt-5.6-sol、provider=openai、effort=max、ephemeral=true、readOnly/network=false。',
    '但thread/start同时回报instruction_source_count=1。严格协议要求任务模型只有原harness语义，因此脚本没有发送turn/start。该instruction source的路径/内容未保存，也没有为通过检查而放宽条件。',
    '',
    '首条结果：失败于推理前隔离门。分类请求数0，答案/正确性/耗时/token均无，工具调用0，答案重试0。',
    '额度：2026-10-05T07:29:01Z查询，codex 7天窗used=88%、derived remaining=12%、reset=2026-10-10T02:34:00Z。没有调用后快照，因为没有推理。',
    '完整40次预算：无法用0个任务样本估算；12%粗粒度剩余额度且bucket到模型的计量关系未知，因此不保守批准余下39次。',
    '',
    '语义差异：app-server为Codex agent transport；CLI 0.149.1没有temperature/max-output-token字段，也没有关闭核心工具声明的字段。web已禁用、cwd隔离、沙箱只读；这些仍不足以证明raw-completion等价。',
    '没有读取/复制凭证，没有登录、权限、配置、原代码/日志修改，没有安装软件。ephemeral helper关闭stdio后正常退出。',
]
(OUT / '结论.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'review': review['status'],
                  'turn_start_requests_sent': result['turn_start_requests_sent'],
                  'instruction_source_count': result['model_echo']['instruction_source_count'],
                  'quota_remaining_percent': report['quota']['latest_weekly_remaining_percent'],
                  'sources_unchanged': checks['all_frozen_sources_unchanged']}, ensure_ascii=False))
