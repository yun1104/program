#!/usr/bin/env python3
"""Export the completed frozen baseline as Markdown and CSV, without resampling."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import project_path, read_json


def percentage(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/eval/qwen35_2b_base_vllm_test_20261026")
    parser.add_argument("--output", default="reports/baseline/qwen35_2b_vllm_test")
    args = parser.parse_args()
    run = project_path(args.run)
    report = read_json(run / "summary.json")
    manifest = read_json(run / "manifest.json")
    records = [read_json(p) for p in sorted((run / "episodes").glob("*.json"))]
    requested = {(identity, trial) for identity in manifest["requested_ids"]
                 for trial in range(manifest["config"]["trials"])}
    actual = {(r["row"]["id"], r["trial"]) for r in records}
    if requested != actual or len(records) != len(requested):
        raise ValueError("Baseline is incomplete or contains extra/duplicate episodes")
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = manifest["config"]
    generation = config["generation"]
    all_values = report["breakdown"]["all"]
    official_pass = sum(r.get("official_pass", False) for r in records)
    strict_pass = sum(r.get("strict_pass", False) for r in records)
    lines = ["# Qwen3.5-2B / vLLM 测试集 Baseline", "",
             f"原始、未经本项目微调的 Qwen3.5-2B；{len(records)} 个测试实例，每题 {config['trials']} 次。",
             "",
             f"官方通过：**{official_pass}/{len(records)}（{percentage(all_values['official_pass_rate'])}）**。",
             f"额外严格通过：**{strict_pass}/{len(records)}（{percentage(all_values['strict_pass_rate'])}）**。",
             f"基础设施错误：{all_values['infrastructure_errors']}，仍计入分母。", "",
             "| 切片 | 样本数 | 官方通过率 | 额外严格通过率 | 平均工具调用 | 平均总 token |", "|---|---:|---:|---:|---:|---:|"]
    keys = ["all"] + [f"mode/P{i}" for i in range(5)] + [f"complexity/C{i}" for i in range(1, 5)] + ["domain/Travel"]
    for key in keys:
        if key not in report["breakdown"]:
            continue
        value = report["breakdown"][key]
        lines.append(f"| {key} | {value['n']} | {percentage(value['official_pass_rate'])} | "
                     f"{percentage(value['strict_pass_rate'])} | {value['mean_tool_calls']:.2f} | {value['mean_tokens']:.1f} |")
    lines += ["", "| 故障模式 | 官方 PRR | 官方 RC |", "|---|---:|---:|"]
    for mode in ("P1", "P2", "P3", "P4"):
        rc = report["official"]["rc"].get(mode)
        lines.append(f"| {mode} | {percentage(report['official']['prr'].get(mode))} | "
                     + ("—" if rc is None else f"{rc:.4f}") + " |")
    terminations = dict(Counter(r.get("termination", "infrastructure_error") for r in records))
    lines += ["", "## 运行口径", "",
              f"- vLLM {manifest['packages'].get('vllm')}，BF16，thinking 关闭，无 JSON 强约束解码。",
              f"- temperature={generation['temperature']}，top_p=1，top_k 关闭；单动作最多 {generation['max_tokens']} token。",
              f"- 上下文上限 {generation['max_context_tokens']} token，交互上限 {manifest['project_config']['max_rounds']} 轮。",
              f"- 模型 revision：`{manifest.get('model_snapshot', {}).get('revision', 'see manifest')}`。",
              f"- 测试集 SHA-256：`{manifest['split_sha256']}`。",
              f"- ToolMaze commit：`{manifest['toolmaze_commit']}`。", "",
              "## 判分说明", "",
              "官方通过包含按任务要求合理终止，不能直接解释为所有事务均完成。额外严格检查要求官方通过、有明确最终回答、全程无动作格式或 schema 错误，也不是完整业务语义证明。",
              "PRR 按独立故障命中统计；RC 仅对实际命中故障的运行计算，数值越低恢复成本越低。Travel 是可与其他领域共存的标签切片。",
              "平均总 token 为各轮输入和输出之和，因此可超过单次上下文上限。每个测试实例仅一次采样，未根据结果挑选重试。", "",
              f"终止类型：`{json.dumps(terminations, ensure_ascii=False)}`。",
              f"含协议错误的轨迹：{sum(r.get('invalid_actions', 0) > 0 for r in records)}；含 schema 错误的轨迹：{sum(r.get('validation_errors', 0) > 0 for r in records)}。", "",
              f"完整配置、权重哈希和原始轨迹：`{run.relative_to(ROOT)}`。"]
    if manifest.get("partitions"):
        lines += ["", f"执行方式：{len(manifest['partitions'])} 个单卡 vLLM 副本，任务分片互斥，合并时校验覆盖全部测试实例。",
                  f"因吞吐从单卡切换为分片，继承全部 {len(manifest.get('inherited_completed_episodes', []))} 条已完成轨迹；"
                  "未完成轨迹重新启动。继承条件核验了输入协议、环境、数据、权重和采样设置，推理调度优化另有记录，未按结果筛选。"]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["slice", *all_values.keys()])
        writer.writeheader()
        for key, values in sorted(report["breakdown"].items()):
            writer.writerow({"slice": key, **values})
    print(output.with_suffix(".md"))
    print(json.dumps(all_values, indent=2))


if __name__ == "__main__":
    main()
