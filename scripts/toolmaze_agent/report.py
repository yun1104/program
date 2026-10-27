"""Build a portable HTML comparison from frozen evaluation outputs."""
import argparse
from html import escape
import json

from .common import project_path, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Evaluation directories")
    parser.add_argument("--output", default="reports/toolmaze_comparison.html")
    args = parser.parse_args()
    rows, details, protocol = [], [], None
    for name in args.runs:
        directory = project_path(name)
        report = read_json(directory / "summary.json")
        manifest = read_json(directory / "manifest.json")
        signature = [manifest["split_sha256"], manifest["requested_ids"],
                     manifest["config"]["trials"], manifest["config"]["generation"],
                     manifest["config"]["seed"], manifest["project_config"]["max_rounds"],
                     manifest["toolmaze_commit"]]
        if protocol is None:
            protocol = signature
        elif protocol != signature:
            raise ValueError("Runs have different tasks, budgets, or upstream versions; cannot compare fairly")
        for cell, value in report["breakdown"].items():
            rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in (
                name, cell, value["n"], f"{value['official_pass_rate']:.1%}",
                f"{value['strict_pass_rate']:.1%}", f"{value['mean_tool_calls']:.2f}",
                f"{value['mean_tokens']:.0f}", value["infrastructure_errors"])) + "</tr>")
        details.append(f"<h2>{escape(name)}</h2><pre>{escape(json.dumps(report['official'], ensure_ascii=False, indent=2))}</pre>")
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ToolMaze 实验比较</title>
<style>body{font:15px system-ui;max-width:1400px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;width:100%}th,td{padding:8px;border:1px solid #ddd;text-align:left}th{background:#eef3f8}pre{white-space:pre-wrap}tr:nth-child(even){background:#fafafa}</style>
<h1>ToolMaze 实验比较</h1><p>官方通过率与额外严格检查分别报告。基础设施错误仍计入分母。结果仅说明受控模拟环境中的表现。</p>
<table><thead><tr><th>实验</th><th>切片</th><th>样本数</th><th>官方通过率</th><th>严格通过率</th><th>工具调用</th><th>Tokens</th><th>基础设施错误</th></tr></thead><tbody>'''
    html += "".join(rows) + "</tbody></table>" + "".join(details) + "</html>"
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
