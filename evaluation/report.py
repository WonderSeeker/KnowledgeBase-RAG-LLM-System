"""出报告：读 run_ragas.py 产出的 JSON，打印指标表；给两份报告就出「改造前 vs 改造后」对比表

用法：
    # 单份报告
    .\\.venv\\Scripts\\python.exe evaluation\\report.py evaluation\\report_baseline.json

    # 两份对比（阶段 3 的验收产物），并把 Markdown 存盘
    .\\.venv\\Scripts\\python.exe evaluation\\report.py ^
        evaluation\\report_baseline.json evaluation\\report_optimized.json --out evaluation\\REPORT.md

    # 只看低分样本（定位是哪些问题拖低了指标）
    .\\.venv\\Scripts\\python.exe evaluation\\report.py evaluation\\report_baseline.json --worst 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
METRIC_CN = {
    "faithfulness": "忠实度（答案是否忠于检索片段）",
    "answer_relevancy": "答案相关性（是否切题）",
    "context_recall": "上下文召回（是否漏检）",
    "context_precision": "上下文精度（噪声多少）",
}


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def delta_text(before, after) -> tuple[str, str]:
    """返回 (差值文本, 百分比变化文本)"""
    if before is None or after is None:
        return "N/A", "N/A"
    diff = after - before
    pct = "N/A" if not before else f"{diff / before * 100:+.1f}%"
    return f"{diff:+.4f}", pct


def describe(report: dict) -> str:
    meta = report.get("meta", {})
    retrieval = meta.get("retrieval", {})
    return (f"{meta.get('name')} | 模式={meta.get('mode')} 裁判={meta.get('judge')}"
            f" | top_k={retrieval.get('top_k')} | 语料分块={retrieval.get('chunk_size')}/"
            f"{retrieval.get('chunk_overlap')} | {meta.get('generated_at')}")


def render(reports: list[dict], labels: list[str], worst: int = 0) -> str:
    lines: list[str] = ["# RAG 评测报告", ""]
    for label, report in zip(labels, reports):
        lines.append(f"- **{label}**：{describe(report)}")

    warnings = [r.get("meta", {}).get("warning") for r in reports]
    warnings = [w for w in warnings if w]
    if warnings:
        # 只用纯文本标记：Windows 控制台是 GBK，emoji 会让 print 抛 UnicodeEncodeError
        lines += ["", f"> **[警告]** {warnings[0]}"]

    # ---- 指标表 ----
    lines += ["", "## 指标对比", ""]
    if len(reports) == 2:
        lines += ["| 指标 | " + labels[0] + " | " + labels[1] + " | Δ | 变化 |", "|---|---|---|---|---|"]
    else:
        lines += ["| 指标 | " + labels[0] + " |", "|---|---|"]
    for name in METRIC_NAMES:
        values = [r.get("metrics", {}).get(name) for r in reports]
        if len(reports) == 2:
            diff, pct = delta_text(values[0], values[1])
            lines.append(f"| {name}<br>{METRIC_CN[name]} | {fmt(values[0])} | {fmt(values[1])} | {diff} | {pct} |")
        else:
            lines.append(f"| {name}<br>{METRIC_CN[name]} | {fmt(values[0])} |")
    lines.append("")

    # ---- 运行开销 ----
    lines += ["## 运行开销", ""]
    header = "| 项 | " + " | ".join(labels) + " |"
    lines += [header, "|" + "---|" * (len(labels) + 1)]
    for key, title in [("avg_n_contexts", "平均检索片段数"), ("avg_total_seconds", "平均单题耗时(s)"),
                       ("run_seconds", "整轮评测耗时(s)")]:
        row = [str(r.get("meta", {}).get(key)) for r in reports]
        lines.append(f"| {title} | " + " | ".join(row) + " |")
    lines.append("")

    # ---- 低分样本 ----
    if worst:
        for label, report in zip(labels, reports):
            items = report.get("per_question", [])
            if not items:
                continue
            scored = []
            for item in items:
                values = [item.get(name) for name in METRIC_NAMES if item.get(name) is not None]
                if values:
                    scored.append((sum(values) / len(values), item))
            scored.sort(key=lambda pair: pair[0])
            lines += [f"## 低分样本（{label} 平均分最低 {min(worst, len(scored))} 条）", ""]
            lines += ["| id | 问题 | 平均分 | 命中片段 | 忠实度 | 召回 |", "|---|---|---|---|---|---|"]
            for avg, item in scored[:worst]:
                lines.append(f"| {item.get('id')} | {item.get('question')} | {avg:.3f} | "
                             f"{item.get('n_contexts')} | {fmt(item.get('faithfulness'))} | "
                             f"{fmt(item.get('context_recall'))} |")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评测报告")
    parser.add_argument("reports", nargs="+", help="run_ragas.py 产出的报告 JSON（1 份=单表，2 份=对比表）")
    parser.add_argument("--out", default=None, help="把 Markdown 写入文件（留空只打印到终端）")
    parser.add_argument("--worst", type=int, default=0, help="额外列出平均分最低的 N 条样本")
    args = parser.parse_args()

    paths = [Path(p) for p in args.reports]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"报告不存在：{path}")

    reports = [load_report(str(p)) for p in paths]
    labels = [r.get("meta", {}).get("name", p.stem) for r, p in zip(reports, paths)]

    modes = {r.get("meta", {}).get("mode") for r in reports}
    judges = {r.get("meta", {}).get("judge") for r in reports}
    if len(modes) > 1 or len(judges) > 1:
        print(f"[警告] 两份报告的 模式/裁判 不一致（mode={modes}, judge={judges}），对比结论无效，请先统一口径。\n")
    if "mock" in modes or "heuristic" in judges:
        print("[警告] 含 MOCK/启发式数据：打分不是 RAGAS，只用于验证流程，不能当作真实效果。\n")

    text = render(reports, labels, worst=args.worst)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"\n[报告] 已写入：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
