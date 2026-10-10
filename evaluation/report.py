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

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
METRIC_CN = {
    "faithfulness": "忠实度（答案是否忠于检索片段）",
    "answer_relevancy": "答案相关性（是否切题）",
    "context_recall": "上下文召回（是否漏检）",
    "context_precision": "上下文精度（噪声多少）",
}

# 指标渲染顺序：先 RAGAS 4 项，再文档级 IR，最后片段级（弱标注）
_IR_KINDS = ("hit_rate@1", "hit_rate@3", "precision@1", "precision@3", "precision@5",
             "lift@1", "lift@3", "lift@5",
             "recall@1", "recall@3", "recall@5", "mrr", "ndcg@5")
# 顺序即"可信度递降"：文档级（非循环） → 对抗性口径（刻意偏袒关键词检索） → 弱标注（与向量检索同源）
PREFERRED_ORDER = (RAGAS_METRICS
                   + [f"doc_{k}" for k in _IR_KINDS]
                   + [f"kw_{k}" for k in _IR_KINDS]
                   + [f"chunk_{k}" for k in _IR_KINDS])

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
    "deterministic": {
        "title": "确定性 IR 指标（**不需要 LLM 裁判**）",
        "algo": {
            "hit_rate": "top-k 里是否至少命中一个正例（0/1，对 k 个 query 取平均）",
            "recall": "top-k 覆盖了正例集合的多大比例（注意：会被语料粒度污染，跨分块配置不可比）",
            "precision": "top-k 里有多大比例是正例（噪声视角；与语料总片段数无关，可跨分块配置比较）",
            "mrr": "第一个正例的排名倒数（1/rank），对「排得够不够前」最敏感",
            "ndcg": "二元相关性的 nDCG@k；正例全部排在最前时为 1.0",
        },
        "note": "只测**检索排序质量**，不含生成质量。`doc_*` 为文档级主口径；"
                "`chunk_*` 为片段级**弱标注**口径（自动生成，非人工标注，阈值见报告 caliber）。",
    },
}


def metric_names_of(reports: list[dict]) -> list[str]:
    """各报告出现过的指标名，按 PREFERRED_ORDER 排序（报告之间指标集可以不同）"""
    present: set[str] = set()
    for report in reports:
        present.update((report.get("metrics") or {}).keys())
    present.discard("n")
    ordered = [name for name in PREFERRED_ORDER if name in present]
    ordered += sorted(present - set(ordered))
    return ordered


def metric_label(name: str) -> str:
    """指标显示名：RAGAS 4 项用固定中文名，IR 指标按 前缀+类型+k 动态生成"""
    if name in METRIC_CN:
        return METRIC_CN[name]
    for prefix, level in (("doc_", "文档级"), ("kw_", "对抗口径"), ("chunk_", "片段级(弱标注)")):
        if name.startswith(prefix):
            kind, _, k = name[len(prefix):].partition("@")
            kind_cn = {"hit_rate": "命中率", "recall": "召回率", "precision": "精度(噪声)",
                       "lift": "lift(相对基准率)", "mrr": "MRR", "ndcg": "nDCG"}.get(kind, kind)
            return f"{level}{kind_cn}{'@' + k if k else ''}"
    return name


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


def caliber_conflicts(reports: list[dict]) -> list[str]:
    """找出各报告之间「口径不一致」的地方：k / 嵌入后端 / 嵌入维度

    这三者任一不同，指标就没有可比性（尤其是嵌入维度不同 = 向量空间不同）。
    把它抽成函数，让终端告警与 Markdown 告警用的是同一套判断。
    """
    metas = [(r.get("meta", {}) or {}).get("retrieval", {}) or {} for r in reports]
    warnings: list[str] = []

    top_ks = sorted({m.get("top_k") for m in metas} - {None})
    if len(top_ks) > 1:
        warnings.append(f"各报告 top_k 不同（{top_ks}）：检索片段数不同时 context_precision / "
                        f"context_recall 不具直接可比性，请先对齐 k。")

    backends = sorted({str(m.get("embedding_backend")) for m in metas
                       if m.get("embedding_backend") is not None})
    dims = sorted({m.get("embedding_dim") for m in metas if m.get("embedding_dim") is not None})
    if len(backends) > 1 or len(dims) > 1:
        warnings.append(f"嵌入口径不同（后端 {backends} / 维度 {dims}）：向量空间不同，"
                        f"指标不可跨行比较。")

    # 检索口径也可能是"看不见的变量"：同一 k 下 candidate_k 不同，粗排池大小就不同，
    # 混合检索/重排的结果会跟着变（实测同一配置 candidate_k=32 -> doc_hit@1 0.7222、=10 -> 0.6111）
    candidate_ks = sorted({m.get("candidate_k") for m in metas if m.get("candidate_k") is not None})
    if len(candidate_ks) > 1:
        warnings.append(f"各报告 candidate_k 不同（{candidate_ks}）：粗排候选池大小不同，"
                        f"会影响检索结果，比较前请先对齐。")

    reranks = sorted({bool(m.get("rerank")) for m in metas})
    if len(reranks) > 1:
        warnings.append(f"各报告的重排开关不同（{[('开' if r else '关') for r in reranks]}）："
                        f"重排会改变排序，比较前请先对齐。")
    return warnings


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
            f"{repeat_text(report)}"
            f" | {meta.get('generated_at')}")


def noise_floor_of(reports: list[dict], metric_name: str) -> tuple[float | None, float | None]:
    """本次对比中该指标可接受的"噪声地板"：各报告实测极差里的最大值

    逐指标优先用报告的 `noise_floor.by_metric`（同一配置重复测出的**该指标**极差）；
    没有该指标的细分数据时退回该报告的总体极差 —— 保守选择：宁可把差异判成噪声，
    也不要凭空宣称发现。
    返回 (地板值, 换算成题数)；参与对比的报告都没测过噪声地板时返回 (None, None)。
    """
    floors: list[tuple[float, float | None]] = []
    for report in reports:
        nf = (report.get("meta", {}) or {}).get("noise_floor") or {}
        if not nf:
            continue
        eval_size = nf.get("eval_size")
        by_metric = nf.get("by_metric") or {}
        value = by_metric.get(metric_name)
        if value is None:
            value = nf.get("value")
        if value is None:
            continue
        items = round(float(value) * eval_size, 1) if eval_size else None
        floors.append((float(value), items))
    if not floors:
        return None, None
    return max(floors, key=lambda pair: pair[0])


def repeat_text(report: dict) -> str:
    """重复测量的显示文本：`×3 次均值（噪声地板 2 题 / 36）`"""
    meta = report.get("meta", {}) or {}
    n = meta.get("repeat_n")
    if not n or int(n) <= 1:
        return ""
    nf = meta.get("noise_floor") or {}
    if nf:
        return f" ×{n} 次均值（实测噪声地板 {nf.get('items')} 题 / {nf.get('eval_size')}）"
    return f" ×{n} 次均值"


def render(reports: list[dict], labels: list[str], worst: int = 0) -> str:
    lines: list[str] = ["# RAG 评测报告", ""]
    for label, report in zip(labels, reports):
        lines.append(f"- **{label}**：{describe(report)}")

    warnings = [r.get("meta", {}).get("warning") for r in reports]
    warnings = [w for w in warnings if w]
    if warnings:
        # 只用纯文本标记：Windows 控制台是 GBK，emoji 会让 print 抛 UnicodeEncodeError
        lines += ["", f"> **[警告]** {warnings[0]}"]

    # 口径校验：k / 嵌入后端 / 维度不一致就没有可比性，必须显式警告
    for conflict in caliber_conflicts(reports):
        lines += ["", f"> **[警告] 口径不一致**：{conflict}"]

    # ---- 指标表（支持 1 份 / 2 份 / N 份：N 份时以第一份为基准算 Δ）----
    lines += ["", "## 指标对比", ""]
    header = "| 指标 | " + " | ".join(labels) + " |"
    separator = "|" + "---|" * (len(labels) + 1)
    if len(reports) >= 2:
        header += " Δ(末-首) | 变化 |"
        separator += "---|---|"
    lines += [header, separator]
    noise_rows: list[tuple[str, float, float, float | None]] = []
    for name in metric_names_of(reports):
        values = [r.get("metrics", {}).get(name) for r in reports]
        row = f"| {name}<br>{metric_label(name)} | " + " | ".join(fmt(v) for v in values) + " |"
        if len(reports) >= 2:
            diff, pct = delta_text(values[0], values[-1])
            mark = ""
            floor, floor_items = noise_floor_of(reports, name)
            numbers = [float(v) for v in values if v is not None]
            if floor is not None and len(numbers) >= 2:
                # ★ 自动拦截：Δ 列只有"末-首"一列，但三列之间有 3 对比较。
                #   所以按**本行最宽的一对**（max-min）判断：连最宽的一对都被噪声盖住，
                #   这一行就无法排序；否则说明行内至少有一对是可分的（哪怕首末那对不可分）。
                widest = max(numbers) - min(numbers)
                head_delta = abs(numbers[-1] - numbers[0])
                if widest <= floor:
                    mark = " **[本行最宽差异也在噪声内，无法排序]**"
                    noise_rows.append((name, widest, floor, floor_items, "widest"))
                elif head_delta <= floor:
                    mark = " **[首末差异在噪声内]**"
                    noise_rows.append((name, head_delta, floor, floor_items, "head"))
            row += f" {diff} | {pct}{mark} |"
        lines.append(row)
    if reports and "n" in (reports[0].get("metrics") or {}):
        lines.append(f"| n（样本数） | " + " | ".join(str(r.get("metrics", {}).get("n")) for r in reports) + " |")
    lines.append("")

    if noise_rows:
        unorderable = [r for r in noise_rows if r[4] == "widest"]
        head_only = [r for r in noise_rows if r[4] == "head"]
        lines += ["> **[警告] 噪声地板**：本次对比使用了实测噪声地板"
                  "（向量腿是 HNSW 近似检索，同一配置跨进程的**候选集合**都会变；"
                  "实测 36 题里 24 题的集合不同，见 CHANGES §4.10.5）。", ""]
        if unorderable:
            lines.append(f"> **{len(unorderable)} 行的组间差异整体落在噪声之内 —— 这些行无法排序：**")
            for name, raw, floor, floor_items, _kind in unorderable:
                unit = f"{floor_items} 题" if floor_items is not None else "—"
                lines.append(f">   - `{name}`：最宽一对 Δ={raw:+.4f} ≤ 地板 {floor:.4f}（{unit}）")
            lines.append("")
        if head_only:
            lines.append(f"> 另有 {len(head_only)} 行的「首末两组」差异在噪声内（行内其它组仍可分）："
                         + "、".join(f"`{r[0]}`" for r in head_only))
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
        if judge == "deterministic":
            # IR 指标：算法按"指标类型"说明；标注口径（文档级 / 片段级弱标注）逐份报告写明
            for kind, text in caliber["algo"].items():
                lines.append(f"| {kind} | {text} |")
            lines.append("")
            for label, report in zip(labels, reports):
                report_meta = report.get("meta", {}) or {}
                if report_meta.get("judge") != judge:
                    continue
                for key, info in (report_meta.get("caliber") or {}).items():
                    extra = ""
                    if key == "chunk" and info.get("threshold") is not None:
                        extra = (f"；相似度阈值 {info['threshold']}，语料 {info.get('chunk_total')} 个片段，"
                                 f"平均每 query {info.get('avg_positive_per_query')} 个正例")
                    lines.append(f"- **{label} / `{key}_*`**：{info.get('text', '')}{extra}")
        else:
            for name in RAGAS_METRICS:
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
            # 排序键优先用「文档级命中@1」（IR 报告）或 faithfulness（RAGAS 报告），都没有就退化为首个可用指标
            sort_key = next((name for name in ("doc_hit@1", "faithfulness")
                             if items[0].get(name) is not None),
                            next((name for name in metric_names_of([report])
                                  if items[0].get(name) is not None), None))
            if sort_key is None:
                continue
            scored = sorted(items, key=lambda item: (item.get(sort_key) is None, item.get(sort_key)))
            lines += [f"## 低分样本（{label} 按 {sort_key} 最低 {min(worst, len(scored))} 条）", ""]
            lines += ["| id | 问题 | 命中片段 | 检索到的文档（按排名） | 命中@1 | MRR |",
                      "|---|---|---|---|---|---|"]
            for item in scored[:worst]:
                ranked = " > ".join(item.get("ranked_docs") or []) or "（无）"
                lines.append(f"| {item.get('id')} | {item.get('question')} | {item.get('n_contexts')} | "
                             f"{ranked} | {fmt(item.get('doc_hit@1'))} | {fmt(item.get('doc_rr'))} |")
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

    for conflict in caliber_conflicts(reports):
        print(f"[警告] 口径不一致：{conflict}\n")

    # 噪声地板：与"口径不一致"同级的下结论前置条件（对比表里也会逐行标注）
    if not any((r.get("meta", {}) or {}).get("noise_floor") for r in reports):
        print("[警告] 参与对比的报告都没测过噪声地板（未用 --repeat）："
              "1~2 题级的差异无法与噪声区分，下结论前请重新用 --repeat 3 跑。\n")

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
