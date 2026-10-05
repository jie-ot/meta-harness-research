"""Audit the stopped E4 batch without making any further model request."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BATCH = OUT / 'batch_observations'
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


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_result(path, pass_number, index):
    value = read_json(path)
    if path.parent.name == 'observation_000_live':
        usage = value.get('token_usage', {}).get('last') or value.get('token_usage', {}).get('total')
        return {'observation_id': 'pass1_item00', 'pass': 1, 'locked_item_index': 0,
                'item_id': value['attempt']['item_id'],
                'status': 'completed' if value['status'] == 'success' else 'failed',
                'failure_reason': value.get('blocker', {}).get('classification') if value.get('blocker') else None,
                'model': value['catalog_evidence']['model'], 'effort': 'max',
                'thread_id': value['model_echo']['thread_id'], 'turn_id': value['model_echo']['turn_id'],
                'thread_echo': value['model_echo'], 'prompt_sha256': value['outgoing_user_input']['prompt_sha256'],
                'prompt_chars': value['outgoing_user_input']['prompt_chars'],
                'response_sha256': value['response']['raw_response_sha256'],
                'extracted_final_answer': value['response']['extracted_final_answer'],
                'format_valid_final_answer': bool(value['response']['extracted_final_answer']),
                'canonical_prediction': value['response']['canonical_prediction'],
                'canonical_target': value['response']['canonical_target'],
                'was_correct': value['response']['was_correct'],
                'elapsed_seconds': value['request_elapsed_seconds'], 'usage': usage,
                'rate_before': value['rate_limits_before']['buckets'][0]['primary'],
                'rate_after': value['rate_limits_after']['buckets'][0]['primary'],
                'tool_item_types': value['tool_item_types'], 'all_item_types': value['all_item_types'],
                'answer_retries': 0, 'request_retries': 0,
                'wrapper_match': value['wrapper_control']['path_and_content_hashes_match_assessment']}
    usage = value.get('token_usage', {}).get('last') if isinstance(value.get('token_usage'), dict) else None
    return {**value, 'usage': usage,
            'wrapper_match': value.get('thread_echo', {}).get('wrapper_hash_match')}


# Preserve the complete first-observation report before replacing final_report.
if (OUT / 'final_report.json').exists() and not (OUT / 'first_observation_report.json').exists():
    (OUT / 'first_observation_report.json').write_bytes((OUT / 'final_report.json').read_bytes())
if (OUT / '结论.txt').exists() and not (OUT / 'first_observation_结论.txt').exists():
    (OUT / 'first_observation_结论.txt').write_bytes((OUT / '结论.txt').read_bytes())
if (OUT / 'output_review.json').exists() and not (OUT / 'first_observation_review.json').exists():
    (OUT / 'first_observation_review.json').write_bytes((OUT / 'output_review.json').read_bytes())

locked = [json.loads(line) for line in (OUT / 'locked_items.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
state = read_json(OUT / 'batch_state.json')
first = normalize_result(OUT / 'observation_000_live/result.json', 1, 0)
second = normalize_result(BATCH / 'pass1_item01/result.json', 1, 1)
third = normalize_result(BATCH / 'pass1_item02/result.json', 1, 2)
actual = [first, second, third]
actual_by_key = {(x['pass'], x['locked_item_index']): x for x in actual}
rows = []
for index, item in enumerate(locked):
    row = {'locked_index': index, 'item_id': item['item_id'], 'target': item['target'],
           'prompt_sha256': item['prompt_sha256']}
    for pass_number in (1, 2):
        result = actual_by_key.get((pass_number, index))
        prefix = f'pass{pass_number}_'
        if result:
            status = result['status']
            row[prefix + 'status'] = status
            row[prefix + 'prediction'] = result.get('extracted_final_answer') or ''
            row[prefix + 'canonical_prediction_json'] = (json.dumps(result['canonical_prediction'], ensure_ascii=False)
                                                           if result.get('canonical_prediction') is not None else '')
            row[prefix + 'correct'] = result.get('was_correct') if status == 'completed' else ''
            row[prefix + 'failure_reason'] = result.get('failure_reason') or ''
            row[prefix + 'tokens'] = result.get('usage', {}).get('totalTokens') if result.get('usage') else ''
            row[prefix + 'elapsed_seconds'] = result.get('elapsed_seconds')
            row[prefix + 'thread_id'] = result.get('thread_id')
            row[prefix + 'turn_id'] = result.get('turn_id')
        else:
            row[prefix + 'status'] = 'not_sent'
            for suffix in ('prediction', 'canonical_prediction_json', 'correct', 'failure_reason',
                           'tokens', 'elapsed_seconds', 'thread_id', 'turn_id'):
                row[prefix + suffix] = ''
    p1, p2 = actual_by_key.get((1, index)), actual_by_key.get((2, index))
    if p1 and p2 and p1['status'] == p2['status'] == 'completed':
        row['correctness_transition'] = ('C' if p1['was_correct'] else 'W') + ('C' if p2['was_correct'] else 'W')
        row['canonical_label_flip'] = p1['canonical_prediction'] != p2['canonical_prediction']
    else:
        row['correctness_transition'] = 'missing'
        row['canonical_label_flip'] = 'missing'
    rows.append(row)
write_csv(OUT / 'two_pass_item_table.csv', rows)

call_rows = []
for value in actual:
    usage = value.get('usage') or {}
    before = value.get('rate_before') or {}
    after = value.get('rate_after') or {}
    call_rows.append({'observation_id': value['observation_id'], 'pass': value['pass'],
                      'locked_item_index': value['locked_item_index'], 'item_id': value['item_id'],
                      'status': value['status'], 'failure_reason': value.get('failure_reason') or '',
                      'model': value['model'], 'effort': value['effort'],
                      'prompt_sha256': value['prompt_sha256'], 'prediction': value.get('extracted_final_answer') or '',
                      'was_correct': value.get('was_correct') if value['status'] == 'completed' else '',
                      'input_tokens': usage.get('inputTokens', ''), 'output_tokens': usage.get('outputTokens', ''),
                      'reasoning_tokens': usage.get('reasoningOutputTokens', ''), 'total_tokens': usage.get('totalTokens', ''),
                      'cached_input_tokens': usage.get('cachedInputTokens', ''),
                      'elapsed_seconds': value.get('elapsed_seconds'),
                      'remaining_before': before.get('remaining_percent_derived', 100 - before.get('used_percent', 100)
                                                    if isinstance(before.get('used_percent'), int) else ''),
                      'remaining_after': after.get('remaining_percent_derived', 100 - after.get('used_percent', 100)
                                                   if isinstance(after.get('used_percent'), int) else ''),
                      'wrapper_match': value.get('wrapper_match'),
                      'tool_items_json': json.dumps(value.get('tool_item_types', [])),
                      'answer_retries': value.get('answer_retries', 0),
                      'request_retries': value.get('request_retries', 0),
                      'thread_id': value.get('thread_id'), 'turn_id': value.get('turn_id')})
write_csv(OUT / 'actual_calls.csv', call_rows)

completed = [x for x in actual if x['status'] == 'completed']
failed = [x for x in actual if x['status'] != 'completed']
known_usage = [x['usage'] for x in completed if x.get('usage')]
pass1_completed = [x for x in actual if x['pass'] == 1 and x['status'] == 'completed']
source_checks = {}
for name, (relative, expected) in EXPECTED.items():
    actual_hash = sha(ROOT / relative)
    source_checks[name] = {'path': relative, 'expected_sha256': expected,
                           'actual_sha256': actual_hash, 'unchanged': actual_hash == expected}
prompt_checks = [{'index': item['index'], 'path': item['prompt_path'],
                  'expected_sha256': item['prompt_sha256'],
                  'actual_sha256': sha(OUT / item['prompt_path']),
                  'unchanged': sha(OUT / item['prompt_path']) == item['prompt_sha256']} for item in locked]
third_events = read_json(BATCH / 'pass1_item02/event_summary.json')
aggregate = {
    'status': 'INCOMPLETE_STOPPED_ON_FIRST_FAILED_TURN',
    'coverage': {'authorized_total_observations': 40, 'turn_start_requests_sent': 3,
                 'completed_with_score': len(completed), 'failed_missing_score': len(failed),
                 'not_sent': 40 - len(actual), 'pass1_completed': len(pass1_completed),
                 'pass1_failed': 1, 'pass1_not_sent': 17,
                 'pass2_completed': 0, 'pass2_failed': 0, 'pass2_not_sent': 20},
    'pass_accuracy': {'pass1': {'correct': sum(x['was_correct'] is True for x in pass1_completed),
                                'denominator_completed': len(pass1_completed),
                                'accuracy_percent_on_completed_only': 100 * sum(x['was_correct'] is True for x in pass1_completed) / len(pass1_completed),
                                'coverage_percent_of_locked_pass': 100 * len(pass1_completed) / 20,
                                'warning': 'incomplete prefix only; not a full-pass estimate'},
                      'pass2': {'correct': 0, 'denominator_completed': 0,
                                'accuracy_percent_on_completed_only': None,
                                'coverage_percent_of_locked_pass': 0,
                                'warning': 'no requests sent'}},
    'paired_repeat_statistics': {'paired_completed_items': 0, 'CC': None, 'CW': None,
                                 'WC': None, 'WW': None, 'canonical_label_flips': None,
                                 'reason': 'second pass was never started'},
    'usage': {'known_token_observations': len(known_usage), 'missing_token_observations': len(actual) - len(known_usage),
              'known_input_tokens': sum(x['inputTokens'] for x in known_usage),
              'known_output_tokens': sum(x['outputTokens'] for x in known_usage),
              'known_reasoning_tokens': sum(x['reasoningOutputTokens'] for x in known_usage),
              'known_total_tokens': sum(x['totalTokens'] for x in known_usage),
              'known_cached_input_tokens': sum(x['cachedInputTokens'] for x in known_usage),
              'elapsed_seconds_all_actual_turns': sum(x['elapsed_seconds'] for x in actual)},
    'quota': {'first_before_remaining_percent': call_rows[0]['remaining_before'],
              'last_after_remaining_percent': call_rows[-1]['remaining_after'],
              'window_duration_minutes': 10080, 'quota_guard_triggered': False,
              'stop_was_failure_at_remaining_percent': call_rows[-1]['remaining_after']},
    'failure': {'observation_id': third['observation_id'], 'item_id': third['item_id'],
                'turn_status': next((x.get('turn_status') for x in third_events if x['method'] == 'turn/completed'), None),
                'error_notification_seen': any(x['method'] == 'error' for x in third_events),
                'error_payload_retained': False,
                'final_text_present': False, 'token_usage_present': False,
                'scored_as_incorrect': False, 'request_or_answer_retried': False,
                'batch_stop_reason': state['stop_reason']},
    'protocol': {'candidate': 'confusion_disambiguation_memory',
                 'model': 'gpt-5.6-sol', 'effort': 'max',
                 'distinct_thread_ids': len({x['thread_id'] for x in actual}),
                 'all_ephemeral': all(x.get('thread_echo', {}).get('ephemeral') is True for x in actual),
                 'all_wrapper_hashes_match': all(x.get('wrapper_match') is True for x in actual),
                 'all_actual_prompt_hashes_match_locked': all(x['prompt_sha256'] == locked[x['locked_item_index']]['prompt_sha256'] for x in actual),
                 'tool_items_across_actual_turns': sorted({t for x in actual for t in x.get('tool_item_types', [])}),
                 'model_substitutions': 0, 'training_runs': 0, 'candidate_generations': 0,
                 'request_retries': 0, 'answer_retries': 0,
                 'client_result_cache_used': False,
                 'provider_cached_input_tokens_known': sum(x['cachedInputTokens'] for x in known_usage),
                 'state_reset': 'new ephemeral thread per actual request; fixed prompt bytes; no learning'},
    'source_recheck': source_checks, 'prompt_recheck': prompt_checks,
    'scientific_limits': ['coverage is the fixed manifest prefix, not a favorable subset',
                          'no paired-repeat statistics are estimable',
                          'no reliable variance, significance, or stopping-threshold conclusion',
                          'old gpt-oss results are not combined with this GPT-5.6-Sol regime'],
}
write_json(OUT / 'aggregate.json', aggregate)
report = {'status': aggregate['status'], 'summary': aggregate,
          'tables': {'twenty_item_two_pass': 'two_pass_item_table.csv', 'actual_calls': 'actual_calls.csv'},
          'logs': {'batch_state': 'batch_state.json', 'failed_observation': 'batch_observations/pass1_item02',
                   'first_observation': 'observation_000_live'},
          'no_further_model_requests_after_failure': True}
write_json(OUT / 'final_report.json', report)
checks = {
    'exact_three_actual_turns': state['turn_start_requests_sent'] + 1 == 3,
    'fixed_prefix_order': [(x['pass'], x['locked_item_index']) for x in actual] == [(1, 0), (1, 1), (1, 2)],
    'two_completed_one_missing': len(completed) == 2 and len(failed) == 1 and third['was_correct'] is None,
    'failed_not_scored_as_incorrect': third['status'] == 'failed' and third['canonical_prediction'] is None,
    'no_second_pass_started': all(x['pass'] == 1 for x in actual),
    'all_model_effort_exact': all(x['model'] == 'gpt-5.6-sol' and x['effort'] == 'max' for x in actual),
    'distinct_ephemeral_threads': aggregate['protocol']['distinct_thread_ids'] == 3 and aggregate['protocol']['all_ephemeral'],
    'wrapper_and_prompt_match': aggregate['protocol']['all_wrapper_hashes_match'] and aggregate['protocol']['all_actual_prompt_hashes_match_locked'],
    'no_tool_items': not aggregate['protocol']['tool_items_across_actual_turns'],
    'zero_retries_substitutions_training_candidates': all(aggregate['protocol'][k] == 0 for k in ('model_substitutions','training_runs','candidate_generations','request_retries','answer_retries')),
    'all_sources_unchanged': all(x['unchanged'] for x in source_checks.values()),
    'all_twenty_prompts_unchanged': all(x['unchanged'] for x in prompt_checks),
    'batch_stopped_and_helper_exited': state['status'] == 'stopped_on_failure' and state['helper_cleanup'] == 'this_batch_app_server_exited_after_stdio_closed',
    'twenty_row_table': len(rows) == 20,
    'paired_statistics_missing': aggregate['paired_repeat_statistics']['paired_completed_items'] == 0,
}
review = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
          'meaning': 'audit consistency pass; experiment remains incomplete'}
write_json(OUT / 'output_review.json', review)
lines = [
    'E4最终状态：因第三个正式turn失败而停止，未重试；实验不完整', '',
    '严格顺序实际覆盖：第一遍manifest索引0、1完成，索引2发出但失败；索引3–19未发；第二遍20题均未发。不是按结果挑选的子集。',
    f"请求总数3；完成并可评分2；失败缺失1；未发送37。完成的两题均正确，因此第一遍已完成prefix准确率2/2=100%，但覆盖仅2/20=10%，不能当作完整第一遍准确率。第二遍accuracy缺失。",
    'CC/CW/WC/WW与label flip均缺失，因为没有任何题完成两遍。',
    f"已知token覆盖2/3个turn：input={aggregate['usage']['known_input_tokens']}，output={aggregate['usage']['known_output_tokens']}，reasoning={aggregate['usage']['known_reasoning_tokens']}，total={aggregate['usage']['known_total_tokens']}，cachedInput={aggregate['usage']['known_cached_input_tokens']}。三个turn可见耗时合计={aggregate['usage']['elapsed_seconds_all_actual_turns']:.3f}秒。",
    '失败题lawbench_3-3_0027：收到error通知，随后turn/completed status=failed；无最终文本、无token usage，故标missing，不计作错误。logger未保存error payload，根因细节不可恢复。',
    f"7天额度从首条前11%剩余到失败题后仍显示11%；未触发2%阈值，停止原因是turn失败。整数显示不代表零成本。",
    '三次均gpt-5.6-sol/max、不同ephemeral thread、同wrapper哈希、同锁定prompt，工具item为0；无重试、替换模型、训练、candidate生成或client结果缓存。第二个完成turn报告3200 cached input token，属于provider输入缓存记录。',
    '原代码、memory、manifest、score及20个prompt哈希均未变。不给方差、显著性、最优停止轮数或完整重复性结论。',
]
(OUT / '结论.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'status': aggregate['status'], 'review': review['status'],
                  'coverage': aggregate['coverage'], 'pass_accuracy': aggregate['pass_accuracy'],
                  'paired_repeat_statistics': aggregate['paired_repeat_statistics'],
                  'usage': aggregate['usage'], 'quota': aggregate['quota'],
                  'failure': aggregate['failure']}, ensure_ascii=False, indent=2))
