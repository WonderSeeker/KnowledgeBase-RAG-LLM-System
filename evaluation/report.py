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

# ★ 同名指标在两种口径下算法完全不同，必须逐条写清楚，否则只看表头必然误读
CALIBER = {
    "heuristic": {
        "title": "离线启发式代理指标（**不是 RAGAS**）",
        "algo": {
            "faithfulness": "答案的字符二元组有多少比例出现在检索片段里（防幻觉，纯字符串重叠）",
            "answer_relevancy": "问题的字符二元组有多少比例被答案覆盖（纯字符串重叠）",
            "context_recall": "标准答案的字符二元组有多少比例被检索片段覆盖（纯字符串重叠）",
            "context_precision": "各检索片段与标准答案的覆盖率取平均（纯字符串重叠）",
        },
        "note": "全部为字符二元组重叠率，**不调用任何模型**，与 RAGAS 官方算法无关；"
                "检索侧在 mock 模式下也是离线替身，因此本表只能证明流程可跑通，**不可作为效果结论**。",
    },
    "ragas": {
        "title": "RAGAS 官方实现",
        "algo": {
            "faithfulness": "把答案拆成若干陈述，逐条用裁判 LLM 判断能否由检索片段推出，取成立比例",
            "answer_relevancy": "用裁判 LLM 由答案反推可能的问题，与真实问题做向量余弦相似度",
            "context_recall": "用裁判 LLM 从标准答案提取陈述，逐条判断是否被检索片段支持，取比例",
            "context_precision": "逐条判断检索片段与问题的相关性，按排名加权（平均精度）",
        },
        "note": "需要裁判模型（见各报告的 judge_model）与 embedding 接口；分数受裁判模型影响，需固定裁判模型再比较。",
    },
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
    method = retrieval.get("method", "?")
    if retrieval.get("offline_stand_in"):
        method += "（离线替身）"
    detail = f"检索={method} top_k={retrieval.get('top_k')}"
    if retrieval.get("candidate_k"):
        detail += f" 候选={retrieval.get('candidate_k')}"
    if retrieval.get("rerank"):
        detail += f" 重排={retrieval.get('rerank_model')}"
    return (f"{meta.get('name')} | 模式={meta.get('mode')} 裁判={meta.get('judge')} | {detail}"
            f" | 语料分块={retrieval.get('chunk_size')}/{retrieval.get('chunk_overlap')}"
            f" | {meta.get('generated_at')}")


def render(reports: list[dict], labels: list[str], worst: int = 0) -> str:
    lines: list[str] = ["# RAG 评测报告", ""]
    for label, report in zip(labels, reports):
        lines.append(f"- **{label}**：{describe(report)}")

    warnings = [r.get("meta", {}).get("warning") for r in reports]
    warnings = [w for w in warnings if w]
    if warnings:
        # 只用纯文本标记：Windows 控制台是 GBK，emoji 会让 print 抛 UnicodeEncodeError
        lines += ["", f"> **[警告]** {warnings[0]}"]

    # 口径校验：k 不同就没有可比性，必须显式警告（context_precision 随 k 增大天然下降）
    top_ks = sorted({(r.get("meta", {}) or {}).get("retrieval", {}).get("top_k") for r in reports} - {None})
    if len(top_ks) > 1:
        lines += ["", f"> **[警告] 口径不一致**：各报告的 top_k 不同（{top_ks}）。"
                      f"检索片段数不同，context_precision / context_recall 不具直接可比性，"
                      f"请先把 k 对齐再下结论。"]

    # ---- 指标表（支持 1 份 / 2 份 / N 份：N 份时以第一份为基准算 Δ）----
    lines += ["", "## 指标对比", ""]
    header = "| 指标 | " + " | ".join(labels) + " |"
    separator = "|" + "---|" * (len(labels) + 1)
    if len(reports) >= 2:
        header += " Δ(末-首) | 变化 |"
        separator += "---|---|"
    lines += [header, separator]
    for name in METRIC_NAMES:
        values = [r.get("metrics", {}).get(name) for r in reports]
        row = f"| {name}<br>{METRIC_CN[name]} | " + " | ".join(fmt(v) for v in values) + " |"
        if len(reports) >= 2:
            diff, pct = delta_text(values[0], values[-1])
            row += f" {diff} | {pct} |"
        lines.append(row)
    lines.append("")

    # ---- 指标口径说明（逐条写清 mock 与 real 下的真实算法）----
    lines += ["## 指标口径说明", ""]
    by_judge: dict[str, list[str]] = {}
    for label, report in zip(labels, reports):
        judge = (report.get("meta", {}) or {}).get("judge") or "heuristic"
        by_judge.setdefault(judge, []).append(label)
    for judge, group in by_judge.items():
        caliber = CALIBER.get(judge, CALIBER["heuristic"])
        suffix = ""
        if judge == "ragas":
            models = {(r.get("meta", {}) or {}).get("judge_model") for r in reports
                      if (r.get("meta", {}) or {}).get("judge") == "ragas"}
            suffix = f"（裁判模型：{'、'.join(sorted(m for m in models if m))}）"
        lines += [f"**{'、'.join(group)} —— 口径：{caliber['title']}**{suffix}", "",
                  "| 指标 | 本口径下的实际算法 |", "|---|---|"]
        for name in METRIC_NAMES:
            lines.append(f"| {name} | {caliber['algo'][name]} |")
        lines += ["", f"> {caliber['note']}", ""]

    if len(by_judge) > 1:
        lines += ["> **[警告]** 同一张表里混了两种口径（" + "、".join(sorted(by_judge)) + "），"
                  "同名指标的算法不同，**行内数值不可互相比较**。", ""]

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

    top_ks = sorted({(r.get("meta", {}) or {}).get("retrieval", {}).get("top_k") for r in reports} - {None})
    if len(top_ks) > 1:
        print(f"[警告] 口径不一致：各报告 top_k = {top_ks}。"
              f"检索片段数不同时 context_precision 不可横向比较，请先对齐 k。\n")

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
