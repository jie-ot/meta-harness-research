"""Write the final Chinese report from independently verified real outputs."""
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pilot.state import atomic_json, read_json, read_jsonl, utc_now

OUT = ROOT / "analysis"
RUNS = ["D0_a", "D0_b", "D100_a", "D100_b"]


def csv_rows(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def fmt(value, decimals=1):
    return f"{float(value):.{decimals}f}"


def injected_report():
    path = OUT / "injected_results.json"
    if not path.exists():
        return []
    result = read_json(path)
    assert result["status"] == "PASS" and result["formal_candidates"] == 4
    candidates = result["candidates"]
    by_slot = {(r["run_id"].removeprefix("D100_injected_"), r["slot"]): r for r in candidates}
    arm_costs = defaultdict(float)
    for request in read_json(ROOT / "external/injected_budget.json")["requests"].values():
        arm_costs[request["run_id"]] += request["cost_usd"]
    assert abs(sum(arm_costs.values()) - result["proposer_total_usd_including_failures"]) < 1e-7
    parts = [
        f"2026年10月5日补测结论：把H0的额外100条反馈正文完整放入提示词后，未观察到相对D0的稳定提升。四个新候选的测试均值为{result['offspring_mean_audit_pp']:.2f}%，D0为{result['D0_offspring_mean_audit_pp']:.2f}%，下降{-result['offspring_mean_delta_vs_D0_pp']:.2f}个百分点。按原评分规则选优并允许保留H0后，两组交付测试均值为{result['selected_mean_audit_pp']:.2f}%，D0为36.00%，下降{-result['selected_mean_delta_vs_D0_pp']:.2f}个百分点；a、b两组分别变化−8.5和+6.0个百分点。a组选中H0，其27.5%来自原先两次冻结测试29%和26%的均值，不是本次新增测试。",
        "补测D100_injected_a/b均从H0（R4B）出发，复用已经运行过的同一批100条额外反馈。两组最终原生优化器会话的首次请求均核验含完整100条正文，包括题目、预测、答案、对错及可见响应；按用户确认排除单列的内部推理字段。完整注入不等于能够证明模型逐条理解或有效利用。沿用原Meta-Harness的分析、原型试验、实现、自检和候选提交流程；每组在自己的上下文中生成A、B两个候选，因此是两个独立组各生成两个，并非一个共享上下文直接生成四个。两组之间的历史和任务模型缓存分别保存，组内仍遵守原来的共享方式。",
        "四个候选在新候选评估开始前全部封存。每个候选从空memory完成200题训练、100题评分和100题外部测试，模型、题单、顺序和每个候选内部的在线训练流程保持不变。经用户同意提高的是候选之间的训练并发，最多四个同时运行；关机时a-B保留到第147题，次日从第148题续跑，已完成阶段和响应缓存复用。没有为本次补测重新生成H0反馈或重跑D0，也没有依据测试成绩修改候选或选优。",
        table(["组 / 候选", "代码名", "评分（%）", "测试（%）", "训练、评分、测试实际API请求数"],
              [(r["run_id"].removeprefix("D100_injected_") + " / " + r["slot"], r["candidate"],
                r["score_pp"], r["audit_pp"], r["solver_api_requests"]) for r in candidates]),
        "a组新候选评分23%和4%，均低于复用的H0评分32%，故保留H0；b组新候选评分36%和31%，H0为27%，故选中label_lexicon_snap_memory，其外部测试为42%。四个新候选全部接受外部测试，子代均值用于分析生成质量，未参与评分选优。",
        table(["首轮配置", "a组交付测试（%）", "b组交付测试（%）", "交付均值（%）", "四个新子代测试均值（%）"],
              [("原D0", 36, 36, 36, result["D0_offspring_mean_audit_pp"]),
               ("原D100（日志可供读取）", 36, 48, result["original_D100_selected_mean_audit_pp"], result["original_D100_offspring_mean_audit_pp"]),
               ("D100_injected（完整正文注入）", "27.5（复用H0）", 42, result["selected_mean_audit_pp"], result["offspring_mean_audit_pp"])]),
        "原D100首轮的原生工具完整返回记录为a组15条、b组0条；本次补测解决了完整100条正文是否进入请求的问题，但成绩没有随可见反馈增多而一致提高。与原D100相比，本次交付均值也从42.00%降至34.75%。这不能证明反馈无用或长上下文有害：只有两组，D0和原D100来自历史运行，外层实际计算量和恢复次数也不相同，尚不是严格预算匹配的因果实验。原先“反馈量值得优先扩展”的判断因此需要收敛，当前证据支持先检查反馈如何转化为可靠实现，不能直接把更多可见反馈等同于更高收益。",
    ]
    a_norm = by_slot[("a", "A")]["saved_response_postprocessing"]
    b_norm = by_slot[("b", "A")]["saved_response_postprocessing"]
    a_format = by_slot[("a", "B")]["saved_response_format_diagnostic"]
    parts.append(f"保存响应的事后诊断显示，a-A的答案归一化存在明显损伤：在能单独恢复原答案的{a_norm['covered']}道无重试测试题中，原响应答对{a_norm['before_correct']}题，转换后仅{a_norm['after_correct']}题，{a_norm['harmed']}道原本正确的答案被改错、纠正{a_norm['helped']}道。相同诊断下，b-A在100题中从原响应答对{b_norm['before_correct']}题变为{b_norm['after_correct']}题，纠正{b_norm['helped']}道、改错{b_norm['harmed']}道。说明具体规则的实现质量直接影响成绩，不能把这两个相反结果都归结为反馈条数。这里固定了已经发生的训练、提示词和响应，只重放末端处理，并非重新训练或运行过的控制消融。")
    parts.append(f"a-B的5%还伴随输出格式问题：100道测试题拆为{a_format['acts']}个行为片段，{a_format['nonempty_first_responses']}次首响应都非空，却全部无法按候选的JSON final_answer提取规则解析，因此触发{a_format['fallback_calls']}次后备调用；最终有{a_format['empty_predictions']}题输出为空，{a_format['predictions_with_terminal_crime_suffix']}题预测含以“罪”结尾的标签，而金标中此类标签为{a_format['targets_with_terminal_crime_suffix']}题。该候选训练、评分、测试合计{by_slot[('a', 'B')]['solver_api_requests']}次真实API请求，明显高于另外三个的400至439次，解释了其耗时较长；这是候选内部的拆分与重试开销。末端校准本身只把答对6题变为5题，不能单独解释全部退化。没有修改候选后再测，亦不能据此预报修复后的成绩。")
    parts.append(f"补测外层Claude Code累计费用为${result['proposer_total_usd_including_failures']:.6f}，其中a组${arm_costs['D100_injected_a']:.6f}、b组${arm_costs['D100_injected_b']:.6f}；所有先前失败、压缩和续接费用都包含在内。两组共10次付费CLI尝试，最终形成两次通过验收的候选提交，不代表10轮实验。任务模型训练、评分、测试另计${result['solver_total_usd']:.6f}，补测合计${result['supplement_total_usd']:.6f}，补测未知费用预留为${result['supplement_reserved_or_unknown_usd']:.2f}。外层未达到用户后来设定的30美元通知门槛。早期上下文处理和续接问题造成额外尝试，这不是一次干净生成四个候选的标准成本；费用均来自已记录返回值或日志估值，仍与现金账单口径区分。")
    parts.append(f"补测独立核验通过：4个候选、12个新增完整阶段、{result['new_prediction_tasks']}次新逐题预测（训练800、评分400、测试400），{result['original_files_unchanged']}个原工程文件哈希未变；封存代码、memory、题单、模型参数、评分选优、缓存和费用记录已交叉检查。独立核验没有模型调用，机器可读结果见injected_results.json。以下保留原实验1、2、3的历史分析；其16个候选、8600次预测、图表、56.810708美元已知费用及2.50美元未知预留均不包含本次补测。原实验与补测合计已知费用约为${read_json(OUT / 'accounting_review.json')['budget']['known_or_estimated_usd'] + result['supplement_total_usd']:.6f}；原实验的80美元上限和补测单独授权的30美元外层门槛分别记录。")
    return parts


def main():
    verification = read_json(OUT / "output_verification.json")
    assert verification["final"] and verification["status"] == "PASS"
    result = read_json(OUT / "analysis.json")
    accounting = read_json(OUT / "accounting_review.json")
    assert result["completed_experiments"] == [1, 2, 3]
    rows = csv_rows("checkpoint_summary.csv")
    costs = csv_rows("checkpoint_cost_detail.csv")
    lookup = {(r["run_id"], int(r["T"])): r for r in rows}
    cost_lookup = {(r["run_id"], int(r["T"])): r for r in costs}
    children = csv_rows("all_candidates.csv")
    version_names={(r["run_id"],r["candidate"]):f"R{r['T']}{r['slot']}" for r in children}
    evidence = csv_rows("optimizer_evidence_access.csv")
    frozen = csv_rows("noise_summary.csv")
    budget = accounting["budget"]
    categories = accounting["cost_categories_usd"]
    noise = result["noise"]
    main_conclusion = result["conclusion"]
    positive_D = any(e["axis"] == "D" and e["signal"] == "positive" for e in result["effects"])
    stable_T = any(e["axis"] == "T" and e["signal"] in {"positive", "negative"} for e in result["effects"])
    if positive_D and not stable_T:
        main_conclusion = "本次更值得继续验证的是反馈量D；追加第二轮尚未带来稳定的最终交付收益。D在两轮配置中的对照达到预设筛查门槛，但只有两次独立重复，且实际读取反馈的数量不一致，尚不能据此确定最优配比。"
    parts = [
        main_conclusion + " 本次只比较追加1轮或2轮、额外反馈0题或100题。实验4按用户要求未运行，因此“能否提前预测未测配比”仍未检验。以下性能均是固定评分集选中版本的外部测试准确率。",
        "数据与模型沿用锁定方案：LawBench 3-3罪名预测，训练200题、固定评分100题、额外反馈池100题、外部测试100题；五百题互不重复，原训练顺序保留。任务模型为gpt-oss-120b，temperature=0、Reasoning: medium；优化器为claude-opus-5-5-code、effort=high。共同起点为R4B，每轮生成2个候选，D0和D100各两条独立轨迹。每个新候选从空memory训练，测试内容与成绩留在优化器工作区之外。",
        f"实验1完成1200次冻结预测，关闭本地响应缓存。R4B两次测试均值S0={noise['S0_pp']:.1f}%；三版本两次测试的最大差值J={noise['J_pp']:.1f}个百分点，预设筛查门槛δ={noise['delta_pp']:.1f}个百分点。两次重复的最大差值只是本次波动参照，不是波动上限或统计显著性阈值。",
        table(["冻结版本", "评分两次（%）", "测试两次（%）"],
              [(label, " / ".join(fmt(r["score_pp"]) for r in frozen if r["version"] == label),
                " / ".join(fmt(r["audit_pp"]) for r in frozen if r["version"] == label))
               for label in ["R4B", "R7A", "R12B"]]),
    ]
    stability = csv_rows("panel_stability.csv")
    parts.append("以第一次完整100题评分排名为参考，固定种子重复抽样1000次，20题和50题出现严格逆序的比例分别为" +
                 "、".join(f"{float(r['any_pair_inversion_rate'])*100:.1f}%" for r in stability) +
                 "。完整100题中R7A与R12B同分，因此“同分状态改变”会显著增加另一项排序变化指标；不能据此说50题比20题更差。完整100题排名也不是真实能力的绝对答案。")
    parts.append("实验2、3的八个正式检查点如下。累计C保留该轨迹实际发生的失败尝试；另列扣除已归档基础设施故障后的实验费用，便于比较资源配比。两列都不包含冻结重评、外部测试和旧memory历史构建费用。后续反馈费用没有回填到第一轮快照。")
    parts.append(table(["运行", "T / N / D", "选中版本", "评分 / 测试S（%）", "累计C（美元）", "扣除基础设施故障后（美元）"],
                       [(r["run_id"], f"{r['T']} / {r['N']} / {r['D']}", version_names.get((r["run_id"],r["selected"]),"H0"),
                         f"{fmt(r['selected_score_pp'])} / {fmt(r['S_pp'])}", fmt(r["C_usd"], 4),
                         fmt(cost_lookup[(r["run_id"], int(r["T"]))]["experimental_C_usd"], 4)) for r in rows]))
    parts.append("R1A表示本条轨迹第一轮的A候选，R2B表示第二轮的B候选；H0是共同起点。完整代码名和全部候选成绩见all_candidates.csv。")
    grid = []
    for t, d in [(1, 0), (1, 100), (2, 0), (2, 100)]:
        group = [lookup[(f"D{d}_{rep}", t)] for rep in ["a", "b"]]
        cost_group = [cost_lookup[(f"D{d}_{rep}", t)] for rep in ["a", "b"]]
        grid.append({"T": t, "D": d, "S": mean(float(r["S_pp"]) for r in group),
                     "C": mean(float(r["C_usd"]) for r in group),
                     "experimental_C": mean(float(r["experimental_C_usd"]) for r in cost_group)})
    parts.append(table(["配置", "平均S（%）", "平均累计C（美元）", "平均实验费用（美元）"],
                       [(f"T={r['T']}, D={r['D']}", fmt(r["S"]), fmt(r["C"], 4), fmt(r["experimental_C"], 4)) for r in grid]))
    grid_lookup = {(r["T"], r["D"]): r for r in grid}
    spending = []
    for d in [0, 100]:
        earlier, later = grid_lookup[(1, d)], grid_lookup[(2, d)]
        spending.append(f"D={d}时，追加第二轮平均多花${later['experimental_C']-earlier['experimental_C']:.4f}，交付测试均值从{earlier['S']:.1f}%变为{later['S']:.1f}%")
    parts.append("把费用与收益放在一起：" + "；".join(spending) + "。是否值得追加不能只看均值，还要看下面两次独立运行的变化是否一致。")
    effect_rows = []
    labels = {"positive": "正向初步信号", "negative": "负向初步信号", "unresolved": "未分辨出稳定作用", "incomplete": "不完整"}
    for effect in result["effects"]:
        if effect["axis"] == "T":
            d = int(effect["fixed"].split("=")[1])
            delta_cost = mean(float(cost_lookup[(f"D{d}_{rep}", 2)]["experimental_C_usd"]) -
                              float(cost_lookup[(f"D{d}_{rep}", 1)]["experimental_C_usd"]) for rep in ["a", "b"])
            title = "多进化一轮，" + effect["fixed"]
        else:
            t = int(effect["fixed"].split("=")[1])
            delta_cost = mean(float(cost_lookup[(f"D100_{rep}", t)]["experimental_C_usd"]) -
                              float(cost_lookup[(f"D0_{rep}", t)]["experimental_C_usd"]) for rep in ["a", "b"])
            title = "多100题反馈，" + effect["fixed"]
        effect_rows.append((title, f"{effect['a_delta_pp']:+.1f} / {effect['b_delta_pp']:+.1f}",
                            f"{effect['mean_delta_pp']:+.1f}", f"{delta_cost:+.4f}", labels[effect["signal"]]))
    parts.append("按事先规则，两次对照同方向且平均绝对变化至少达到δ，才记为初步信号。a、b是独立运行的重复编号；只有两次对照，不构成统计显著性证明。费用差来自各次实际生成、训练和反馈开销，不能当成单独购买100题反馈的固定价格。")
    parts.append(table(["对照", "a / b变化（百分点）", "平均变化", "平均实验费用变化（美元）", "预设筛查结果"], effect_rows))
    parts.append("为区分候选生成与评分选优，下面列出每轮新生成两个子代的平均测试成绩。该列用于诊断，不参与选择交付版本。逐题增加答对、原本答对变错及净变化见paired_changes.csv。")
    parts.append(table(["运行", "首轮子代均值（%）", "第二轮子代均值（%）", "交付S：首轮→第二轮（%）"],
                       [(run, lookup[(run, 1)]["offspring_mean_audit_pp"], lookup[(run, 2)]["offspring_mean_audit_pp"],
                         f"{fmt(lookup[(run,1)]['S_pp'])} → {fmt(lookup[(run,2)]['S_pp'])}") for run in RUNS]))
    for d in [0,100]:
        changes=[]
        for rep in ["a","b"]:
            run=f"D{d}_{rep}"
            earlier,later=lookup[(run,1)]["offspring_mean_audit_pp"],lookup[(run,2)]["offspring_mean_audit_pp"]
            if earlier and later:changes.append(float(later)-float(earlier))
        if len(changes)==2:
            parts.append(f"D={d}时，新子代平均测试成绩从首轮到第二轮的变化分别为{changes[0]:+.1f}、{changes[1]:+.1f}个百分点，平均{mean(changes):+.2f}个百分点。这是生成质量的描述性结果；主结果仍按评分选中的交付版本计算。")
    for run in RUNS:
        first, second = lookup[(run, 1)], lookup[(run, 2)]
        if float(second["offspring_mean_audit_pp"]) > float(first["offspring_mean_audit_pp"]) and float(second["S_pp"]) <= float(first["S_pp"]):
            best_child = max(float(r["audit_pp"]) for r in children if r["run_id"] == run and int(r["T"]) == 2)
            if best_child <= float(first["S_pp"]):
                parts.append(f"{run}说明了两种收益为何不同：第二轮子代平均成绩为{float(second['offspring_mean_audit_pp']):.1f}%，高于首轮的{float(first['offspring_mean_audit_pp']):.1f}%；但第二轮最好子代的测试成绩也只有{best_child:.1f}%，仍未超过首轮选中版本的{float(first['S_pp']):.1f}%。因此该条轨迹最终S没有上升，不能归咎于评分漏选了更好的第二轮候选。这里的测试比较只用于事后诊断。")
    frontier = [r for r in grid if not any(q["experimental_C"] <= r["experimental_C"] and q["S"] >= r["S"] and
                (q["experimental_C"] < r["experimental_C"] or q["S"] > r["S"]) for q in grid)]
    parts.append("扣除独立归档的基础设施故障后，已测四种配置的平均费用—成绩非支配集合为：" +
                 "；".join(f"T={r['T']}、D={r['D']}（S={r['S']:.1f}%，C=${r['experimental_C']:.4f}）" for r in frontier) +
                 "。这只表示本次已测配置中的取舍，不是相同预算下的严格优劣证明，也不是全局最优方案。主费用口径的非支配结果仍保存在grid_summary.csv。")
    visible_rows = []
    for run in RUNS:
        for t in [1, 2]:
            group = [r for r in evidence if r["run_id"] == run and int(r["T"]) == t and r["phase"] == "feedback"]
            visible_rows.append((run, t, sum(int(r["available_items"]) for r in group),
                                 sum(int(r["complete_diagnostic_rows_returned"]) for r in group),
                                 sum(int(r["failed_reads"]) for r in group)))
    parts.append("D表示额外反馈题数，不能直接等同于优化器实际阅读的题数。根据原生工具返回记录，额外反馈的可见情况如下；“完整返回”是成功返回了输入、预测、答案和对错的记录数，不能证明优化器确实利用了每条信息。不同来源候选对同一道题的执行日志分别计数，故第二轮累计可提供300条。")
    parts.append(table(["运行", "生成轮次", "可提供反馈记录", "完整返回记录", "读取失败次数"], visible_rows))
    parts.append("D100_b第一轮曾因长路径拼写错误而未成功读取额外反馈，仍读取了合法的训练或评分历史。本次没有为促使其使用反馈而重新生成候选。因此D对照测量的是“允许并提供额外反馈”的实际流程效果，无法单独分离日志内容质量、阅读量和优化器利用方式的作用。")
    parts.append(f"正式实验共{result['nominal_new_candidates']}个新候选，其中有效{result['valid_new_candidates']}个，正式优化器会话{result['proposer_sessions']}次；另有{accounting['archived_setup_proposer_attempts']}次基础设施故障期间的优化器尝试及其生成的{len(accounting['archived_setup_generated_candidates'])}个候选，总计{accounting['actual_paid_proposer_attempts']}次付费尝试。失败尝试单独归档，不作为独立实验重复或候选质量样本，其费用全部保留在总预算中。已完成阶段共记录{sum(verification['phase_item_counts'].values())}次逻辑题目预测；学习内部调用、重试和失败尝试另计。")
    parts.append(table(["费用类别", "美元"], [("正式搜索", fmt(categories.get("experiment_search", 0), 6)),
                   ("前期基础设施故障", fmt(categories.get("setup_failure", 0), 6)),
                   ("冻结重评", fmt(categories.get("frozen_measurement", 0), 6)),
                   ("正式候选外部测试", fmt(categories.get("external_test", 0), 6)),
                   ("累计已报告或日志估值", fmt(budget["known_or_estimated_usd"], 6)),
                   ("未知费用保守预留", fmt(budget["reserved_or_unknown_usd"], 6))]))
    parts.append(f"总预算上限为80美元。费用来自服务商返回值及明确标记的CLI日志估值，不能当作现金账单。此前{accounting['nonbillable_credit_rejections']}次HTTP402额度检查拒绝单列保留；无金额的中断或失败请求保留预算预留，不虚构为免费。旧R4B历史记录把训练与旧50题验证合在同一笔$0.2204中，无法精确拆出纯memory构建费用，该笔历史成本不计入本次C。")
    ledger = [r for r in read_jsonl(ROOT / "external/cost_ledger.jsonl")
              if not r.get("run_id", "").startswith("D100_injected_")]
    started = [datetime.fromisoformat(r["timestamp_utc"]) for r in ledger if r["event"].endswith("_started")]
    ended = [datetime.fromisoformat(r["timestamp_utc"]) for r in ledger if r["event"].endswith("_result")]
    train_times, interrupted = [], []
    for p in (ROOT / "runs").glob("*/text_classification/history/*/train/result.json"):
        if p.relative_to(ROOT / "runs").parts[0] not in RUNS:
            continue
        if (p.parent / "resume_accounting_review.json").exists():
            interrupted.append(p.relative_to(ROOT).as_posix())
        else:
            train_times.append(read_json(p)["runtime_seconds"])
    durations = [read_json(ROOT / "external/control" / run / f"proposer_round{t}/result.json")["wall_seconds"] for run in RUNS for t in [1, 2]]
    timing = {"generated_at_utc": utc_now(), "first_recorded_request_utc": min(started).isoformat(),
              "last_recorded_result_utc": max(ended).isoformat(),
              "first_recorded_request_Asia_Shanghai": min(started).astimezone(timezone(timedelta(hours=8))).isoformat(),
              "last_recorded_result_Asia_Shanghai": max(ended).astimezone(timezone(timedelta(hours=8))).isoformat(),
              "calendar_span_hours_including_interruptions": (max(ended)-min(started)).total_seconds()/3600,
              "training_stages_with_complete_active_timing": len(train_times), "training_median_minutes": median(train_times)/60 if train_times else None,
              "formal_proposer_median_minutes": median(durations)/60,
              "partial_training_duration_missing": interrupted}
    atomic_json(OUT / "timing_summary.json", timing)
    parts.append(f"从首条已记录请求到末条结果的日历跨度为{timing['calendar_span_hours_including_interruptions']:.2f}小时，包含充值等待、程序修复和暂停。{len(train_times)}个完整计时训练阶段的200题训练中位耗时为{timing['training_median_minutes']:.1f}分钟，正式优化器会话中位耗时为{timing['formal_proposer_median_minutes']:.1f}分钟。另有{len(interrupted)}个阶段缺少中断前一段活动时长，具体路径列于timing_summary.json；这些阶段未计入训练时长中位数。阶段原始耗时和费用保留在stage_usage.csv及各阶段result.json。")
    parts.append("执行修复与例外均保留证据：早期Windows子进程受限和原始日志行过长导致的尝试单独归档；原生CLI现已通过真实工具读取与边界验收。两个仅因注释或通用标签表标题而被误拒的候选恢复了评测，其中一个只清理了文档中的基准名称，执行AST不变。用户随后明确同意忽略任务格式提示问题，原D0_a候选从已有第36步状态续跑，代码未变，也未补生成候选。被替代的无效标记和临时第一轮快照均归档；恢复决定在该候选测试前作出，最终第一轮快照在第二轮反馈和生成之前固定。")
    parts.append("第二轮另有三项运行恢复：原型文件带_proto/_final后缀而被严格文件名检查误拒，已凭原来的成功执行记录通过验收；一次检查点替换失败后，从已保存的第103题续跑，第104题命中本组缓存；随后两个进程退出，其中D0_a记录MemoryError，分别从55题和3题的检查点恢复。已有候选、模型与题单均保留，未补生成候选。没有记录到响应的请求按未知费用保留预算预留。")
    pause_path = ROOT / "external/user_pause.json"
    if pause_path.exists():
        pause = read_json(pause_path)
        saved = "、".join(f"{case['run']}的{case['candidate']}已完成{case['committed_training_steps']}题" for case in pause["cases"])
        parts.append("用户关机前另行要求暂停，确认实验进程全部停止后保留了检查点：" + saved + "。收到继续指令后沿用相同代码、题单和本组缓存恢复；两个没有收到响应的在途请求各保留0.25美元未知费用预留。这次关机间隔计入日历跨度，不计作模型活动耗时，未因暂停补生成候选。")
    parts.append(f"独立复核通过：{verification['original_files_unchanged']}个原工程文件未改，{verification['completed_stages']}个已完成阶段的结果由逐题记录重算一致，四条轨迹八个选择与成本快照可追溯，题单、模型参数、缓存目录和候选代码哈希符合记录。未运行D50，也未把任何外部测试结果提供给优化器。此处采用工具访问限制与独立目录，未使用操作系统级隔离。SDK响应快照保存了完整choices和usage，不包含完整HTTP封包或响应头。")
    parts.append("本次N=2T，只能研究固定每轮两个候选时的串行轮数与反馈配比，不能分别识别T和N。两轮、单任务、每配置两次重复不足以拟合或验证普适scaling law、长期最优停止点或跨任务泛化。下一步是否值得扩展应由上述T/D信号与实际反馈利用情况决定；未测配比的预测能力仍需要预先冻结预测后的独立验证。")
    parts.append("图见frozen_repeats.png、cost_performance.png、comparable_cost_performance.png和round_gain.png；数值见checkpoint_summary.csv、checkpoint_cost_detail.csv、effects.csv、all_candidates.csv、optimizer_evidence_access.csv和paired_changes.csv。external/cost_ledger.jsonl为完整费用台账，output_verification.json为独立结果复核记录。")
    supplement = injected_report()
    if supplement:
        parts[0] = "原实验1、2、3的历史结论（补测前）：" + parts[0]
    (OUT / "给导师的结论.md").write_text("\n\n".join(supplement + parts) + "\n", encoding="utf-8")
    print({"report": str(OUT / "给导师的结论.md"), "completed_experiments": [1, 2, 3], "new_model_calls": 0})


if __name__ == "__main__":
    main()
