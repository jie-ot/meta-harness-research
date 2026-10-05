"""Finalize the one authorized E4 live observation; no model connection."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OBS = OUT / 'observation_000_live'
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


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


# Preserve the earlier strict-gate report as execution history before replacing
# the user-facing final report with the authorized live result.
if (OUT / 'final_report.json').exists() and not (OUT / 'pre_inference_gate_report.json').exists():
    (OUT / 'pre_inference_gate_report.json').write_bytes((OUT / 'final_report.json').read_bytes())
if (OUT / '结论.txt').exists() and not (OUT / 'pre_inference_gate_结论.txt').exists():
    (OUT / 'pre_inference_gate_结论.txt').write_bytes((OUT / '结论.txt').read_bytes())
if (OUT / 'output_review.json').exists() and not (OUT / 'pre_inference_gate_review.json').exists():
    (OUT / 'pre_inference_gate_review.json').write_bytes((OUT / 'output_review.json').read_bytes())

result = json.loads((OBS / 'result.json').read_text(encoding='utf-8'))
events = json.loads((OBS / 'event_summary.json').read_text(encoding='utf-8'))
wrapper = json.loads((OUT / 'instruction_source_assessment.json').read_text(encoding='utf-8'))
tool_precheck = json.loads((OUT / 'tool_feature_precheck.json').read_text(encoding='utf-8'))
lock = json.loads((OUT / 'lock_manifest.json').read_text(encoding='utf-8'))
items = [json.loads(x) for x in (OUT / 'locked_items.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
source_checks = {}
for name, (relative, expected) in EXPECTED.items():
    actual = sha(ROOT / relative)
    source_checks[name] = {'path': relative, 'expected_sha256': expected,
                           'actual_sha256': actual, 'unchanged': actual == expected}
usage = result['token_usage']['last']
all_prompt_chars = sum(x['prompt_chars'] for x in items)
remaining_prompt_chars = 2 * all_prompt_chars - items[0]['prompt_chars']
scaled_remaining_input = round(usage['inputTokens'] * remaining_prompt_chars / items[0]['prompt_chars'])
same_output_remaining = usage['outputTokens'] * 39
planning_same_shape = scaled_remaining_input + same_output_remaining
planning_buffered = round(planning_same_shape * 1.5)

def weekly(snapshot):
    if not snapshot:
        return None
    for bucket in snapshot['buckets']:
        if bucket['bucket_map_key'] == 'codex':
            return bucket['primary']
    return None

before, after = weekly(result['rate_limits_before']), weekly(result['rate_limits_after'])
report = {
    'status': 'FIRST_FORMAL_OBSERVATION_SUCCEEDED',
    'experiment_scope': 'E4 new GPT-5.6-Sol/Max regime only; old gpt-oss results are provenance checks and are not pooled',
    'observation': {'index': 0, 'item_id': result['attempt']['item_id'],
                    'model': result['catalog_evidence']['model'], 'effort': 'max',
                    'thread_id': result['model_echo']['thread_id'], 'turn_id': result['model_echo']['turn_id'],
                    'prompt_sha256': result['outgoing_user_input']['prompt_sha256'],
                    'prompt_chars': result['outgoing_user_input']['prompt_chars'],
                    'raw_response_path': 'observation_000_live/response.txt',
                    'raw_response_sha256': result['response']['raw_response_sha256'],
                    'extracted_final_answer': result['response']['extracted_final_answer'],
                    'canonical_prediction': result['response']['canonical_prediction'],
                    'canonical_target': result['response']['canonical_target'],
                    'was_correct': result['response']['was_correct'],
                    'elapsed_seconds': result['request_elapsed_seconds'],
                    'token_usage': result['token_usage'], 'answer_retries': 0,
                    'tool_items': result['tool_item_types']},
    'visible_model_evidence': {'catalog': result['catalog_evidence'],
                               'thread_start_echo': result['model_echo'],
                               'settings_notification': result['settings_echo_after_turn_start'],
                               'note': 'catalog and thread echo establish this E4 request configuration; they do not identify earlier chat/model runs'},
    'wrapper_control': {'source_type': wrapper['sources'][0]['source_type'],
                        'source_content_sha256': wrapper['sources'][0]['content_sha256'],
                        'source_bytes': wrapper['sources'][0]['bytes'],
                        'same_source_hash_confirmed_on_actual_thread': result['wrapper_control']['path_and_content_hashes_match_assessment'],
                        'task_data_detected': wrapper['sources'][0]['task_data_detected'],
                        'personalization_marker_detected': wrapper['sources'][0]['personalization_marker_detected'],
                        'raw_instruction_text_or_path_saved': False,
                        'base_instruction': 'Reasoning: medium (historical transport string; not the Max setting)',
                        'developer_instruction': 'empty',
                        'ephemeral_new_thread_per_observation': True},
    'tool_isolation': {'official_temporary_feature_states': tool_precheck['effective_states'],
                       'web_search': tool_precheck['web_search'],
                       'actual_tool_items': result['tool_item_types'],
                       'residual_guard_triggered': False,
                       'note': 'No tool action was initiated, so no denial event was needed. Disables were process-local and did not change persistent config.'},
    'quota': {'before': result['rate_limits_before'], 'after': result['rate_limits_after'],
              'weekly_used_percent_before': before['used_percent'],
              'weekly_used_percent_after': after['used_percent'],
              'weekly_remaining_percent_before': before['remaining_percent_derived'],
              'weekly_remaining_percent_after': after['remaining_percent_derived'],
              'displayed_delta_percentage_points': after['used_percent'] - before['used_percent'],
              'interpretation': 'unchanged integer percentage is a coarse display, not evidence of zero cost; bucket-to-model accounting is unspecified'},
    'remaining_39_budget': {'verdict': 'NOT_CONSERVATIVELY_ESTABLISHED; DO NOT AUTO-RUN',
                            'observed_first_turn_tokens': usage['totalTokens'],
                            'remaining_prompt_chars': remaining_prompt_chars,
                            'same_input_density_plus_same_output_point_projection_tokens': planning_same_shape,
                            'one_point_five_times_planning_scenario_tokens': planning_buffered,
                            'limitations': 'not a quota conversion or upper bound; only one output sample, model reasoning/output varies, weekly bucket capacity unknown',
                            'weekly_remaining_display_after': after['remaining_percent_derived'],
                            'remaining_requests_started': 0},
    'frozen_inputs': {'selection_rule': lock['selection_rule'], 'item_count': 20,
                      'selected_item_ids_sha256': lock['selected_item_ids_sha256'],
                      **lock['frozen_hashes']},
    'source_recheck': source_checks,
    'semantic_differences': [
        'This is a fixed official Codex wrapper plus the exact R4B rendered prompt, not a raw completion interface.',
        'CLI 0.149.1 exposes no temperature or max-output-token field; E4 does not claim old temperature=0 transport equivalence.',
        'Reasoning: medium is retained only as the historical base instruction; actual model effort is independently set and echoed as max.',
        'The fixed application-home instruction source is part of the controlled wrapper.',
    ],
    'call_and_log_checks': {'turn_start_requests_sent': result['turn_start_requests_sent'],
                            'turn_status': result['response']['turn_status'],
                            'agent_message_count': result['response']['agent_message_count'],
                            'event_methods': sorted({x['method'] for x in events}),
                            'actual_item_types': result['all_item_types'],
                            'cached_input_tokens': usage['cachedInputTokens'],
                            'cache_write_input_tokens': usage.get('cacheWriteInputTokens'),
                            'client_result_cache_used': False,
                            'helper_cleanup': result['helper_cleanup']},
    'execution_history': {'initial_over_strict_gate': 'stopped after thread/start with zero turns; superseded after wrapper audit',
                          'prompt_byte_repair': 'Windows newline materialization fixed before any turn; all 20 exact byte hashes revalidated',
                          'actual_turns_total': 1, 'answer_retries': 0},
    'safety': {'remaining_39_started': False, 'original_sources_modified': False,
               'credentials_read_or_copied': False, 'login_or_global_settings_changed': False,
               'software_installed': False, 'credit_reset_or_paid_API_used': False},
}
write_json(OUT / 'final_report.json', report)
checks = {
    'live_result_success': result['status'] == 'success' and result['response']['turn_status'] == 'completed',
    'exact_model_and_max': result['catalog_evidence']['model'] == result['model_echo']['model'] == 'gpt-5.6-sol' and result['model_echo']['reasoning_effort'] == 'max',
    'one_actual_turn_only': result['turn_start_requests_sent'] == 1,
    'prompt_exactly_locked': result['outgoing_user_input']['prompt_sha256'] == items[0]['prompt_sha256'],
    'fixed_wrapper_reverified_and_task_clean': result['wrapper_control']['path_and_content_hashes_match_assessment'] and not wrapper['sources'][0]['task_data_detected'],
    'all_official_tool_features_disabled': all(v is False for v in tool_precheck['effective_states'].values()),
    'no_actual_tool_items': not result['tool_item_types'],
    'no_client_or_provider_input_cache_reported': usage['cachedInputTokens'] == usage.get('cacheWriteInputTokens', 0) == 0,
    'raw_response_hash_matches': sha(OBS / 'response.txt') == result['response']['raw_response_sha256'],
    'transition_correctness_consistent': result['response']['canonical_prediction'] == result['response']['canonical_target'] and result['response']['was_correct'] is True,
    'all_original_sources_unchanged': all(x['unchanged'] for x in source_checks.values()),
    'quota_window_is_weekly': before['window_duration_minutes'] == after['window_duration_minutes'] == 10080,
    'remaining_39_not_started': not report['safety']['remaining_39_started'],
    'helper_exited': result['helper_cleanup'] == 'this_ephemeral_app_server_exited_after_stdio_closed',
}
review = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
          'original_project_tests_run': False, 'additional_model_turns_run': 0}
write_json(OUT / 'output_review.json', review)
lines = [
    'E4首条真实分类链路验证：成功；余下39条未启动', '',
    '条件：固定Codex wrapper + R4B rendered prompt，gpt-5.6-sol/max，每条新ephemeral线程。本实验属于新模型regime，不与旧gpt-oss分数合并。',
    f"首条：{result['attempt']['item_id']}；prompt SHA-256={result['outgoing_user_input']['prompt_sha256']}；30,802字符。",
    f"结果：{result['response']['extracted_final_answer']}；canonical={result['response']['canonical_prediction']}；target={result['response']['canonical_target']}；correct={result['response']['was_correct']}。",
    f"耗时：{result['request_elapsed_seconds']}秒。token：input={usage['inputTokens']}，output={usage['outputTokens']}，reasoning={usage['reasoningOutputTokens']}，total={usage['totalTokens']}，cachedInput={usage['cachedInputTokens']}。",
    '',
    '模型证据：完整目录含gpt-5.6-sol及max；thread/start回显model=gpt-5.6-sol、provider=openai、effort=max、ephemeral=true。Reasoning: medium只是保留的旧system文本，不代表当前effort；当前effort由max字段单独设置。',
    f"wrapper：Codex应用目录instruction source，{wrapper['sources'][0]['bytes']}字节，content SHA-256={wrapper['sources'][0]['content_sha256']}；实际thread重新核对相同。未检测到锁定题目ID、target、prompt片段、R4B/LawBench/历史模型标记或个性化标记。未保存原文/路径。",
    '工具隔离：11个官方临时feature开关均effective=false，web disabled；实际item中无工具动作。没有更改持久配置。',
    '',
    f"额度前后：7天窗均used={before['used_percent']}%、remaining={before['remaining_percent_derived']}%；整数显示未变化不能解释为零成本。查询时间见final_report.json。",
    f"预算：余下39条按首条输入密度和同输出的点估计约{planning_same_shape:,} token，1.5倍规划场景约{planning_buffered:,} token；这不是上界或额度换算。仅余{after['remaining_percent_derived']}%且bucket容量未知，因此不保守批准自动运行。",
    '',
    '原代码、memory、manifest和score哈希复核未变；client/result cache未使用，reported cached input=0。答案没有重试。',
]
(OUT / '结论.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'review': review['status'],
                  'prediction': result['response']['extracted_final_answer'],
                  'correct': result['response']['was_correct'],
                  'elapsed_seconds': result['request_elapsed_seconds'],
                  'token_usage': usage, 'quota_before': before, 'quota_after': after,
                  'remaining_39_point_projection_tokens': planning_same_shape,
                  'remaining_39_buffered_planning_tokens': planning_buffered}, ensure_ascii=False, indent=2))
