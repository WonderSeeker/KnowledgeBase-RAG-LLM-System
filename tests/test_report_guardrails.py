"""报告层护栏（evaluation/report.py）

这是本项目最值钱的一层：把「1 题级差异」自动标成"无法排序"，
让过度归因在**下结论之前**就被拦住。所以这里要钉死两件事：

  1) `noise_floor_of` 的口径：逐指标优先（by_metric）、跨报告取最大、没有返回 None
  2) `render` 的两级标注：整行不可排序 / 仅首末噪声内（按"最宽的一对"判断，
     因为三列之间有 3 对比较，只看 Δ(末-首) 会把中间列藏起来）

另：口径不一致拦截（k / candidate_k / 后端+维度 / 重排）也在这里钉死。
"""

from __future__ import annotations

import pytest

import report


def _report(name: str, metrics: dict, *, top_k: int = 4, candidate_k: int = 10,
            backend: str = "local", dim: int = 512, rerank: bool = False,
            floor: float | None = None, by_metric: dict | None = None,
            repeat_n: int = 3, eval_size: int = 36) -> dict:
    """造一份最小可用的报告 JSON（字段形状与 run_ir.py 产出的一致）"""
    meta = {
        "name": name,
        "mode": "ir",
        "judge": "deterministic",
        "generated_at": "2026-10-11 00:00:00",
        "warning": "测试用报告",
        "retrieval": {
            "method": "vector", "top_k": top_k, "candidate_k": candidate_k,
            "rerank": rerank, "embedding_backend": backend, "embedding_dim": dim,
            "chunk_size": 300, "chunk_overlap": 30,
        },
    }
    if floor is not None or by_metric is not None:
        meta["repeat_n"] = repeat_n
        meta["noise_floor"] = {
            "value": floor,
            "items": None if floor is None else round(floor * eval_size, 1),
            "eval_size": eval_size,
            "by_metric": by_metric or {},
        }
    return {"meta": meta, "metrics": metrics}


def _row(text: str, metric: str) -> str:
    """取出表格里某个指标那一行（断言要落到行上，不能整篇 contains）"""
    for line in text.splitlines():
        if line.startswith("| ") and metric in line.split("<br>")[0]:
            return line
    raise AssertionError(f"表格里没有 {metric} 这一行")


# --------------------------------------------------------------------------- noise_floor_of
def test_noise_floor_prefers_per_metric_spread():
    """逐指标极差优先 —— 它与该指标同量纲（lift 是 4~8，hit_rate 是 0~1，不能混用）"""
    report_ = _report("r", {"doc_hit_rate@1": 0.5}, floor=0.5, by_metric={"doc_hit_rate@1": 0.03})
    value, items = report.noise_floor_of([report_], "doc_hit_rate@1")
    assert value == pytest.approx(0.03)
    assert items == pytest.approx(1.1)      # 0.03 * 36 → 换算成"题数"，可操作单位


def test_noise_floor_falls_back_to_headline_value():
    report_ = _report("r", {"doc_mrr": 0.7}, floor=0.05, by_metric={"doc_hit_rate@1": 0.01})
    value, items = report.noise_floor_of([report_], "doc_mrr")
    assert value == pytest.approx(0.05)     # 该指标没有细分数据 → 退回总体值（保守）
    assert items == pytest.approx(1.8)


def test_noise_floor_takes_max_across_reports():
    """一份报告说 0.02、另一份说 0.07 → 取 0.07（宁可把差异判成噪声）"""
    a = _report("a", {"doc_hit_rate@1": 0.5}, floor=0.07, by_metric={"doc_hit_rate@1": 0.02})
    b = _report("b", {"doc_hit_rate@1": 0.5}, floor=0.07, by_metric={"doc_hit_rate@1": 0.07})
    assert report.noise_floor_of([a, b], "doc_hit_rate@1")[0] == pytest.approx(0.07)


def test_noise_floor_is_none_without_measurement():
    """没测过噪声地板（单次运行）→ 返回 None，此时不做任何"无法排序"标注"""
    assert report.noise_floor_of([_report("r", {"doc_hit_rate@1": 0.5})], "doc_hit_rate@1") == (None, None)


# --------------------------------------------------------------------------- repeat_text
def test_repeat_text_is_empty_for_single_run():
    assert report.repeat_text(_report("r", {})) == ""


def test_repeat_text_shows_n_and_items():
    text = report.repeat_text(_report("r", {}, floor=0.083, by_metric={}, repeat_n=3))
    assert "×3" in text and "题" in text


# --------------------------------------------------------------------------- render 的两级标注
def test_render_marks_row_when_widest_pair_is_within_floor():
    a = _report("a", {"doc_hit_rate@1": 0.50, "doc_mrr": 0.60},
                floor=0.03, by_metric={"doc_hit_rate@1": 0.03, "doc_mrr": 0.03})
    b = _report("b", {"doc_hit_rate@1": 0.52, "doc_mrr": 0.70},
                floor=0.03, by_metric={"doc_hit_rate@1": 0.03, "doc_mrr": 0.03})
    text = report.render([a, b], ["a", "b"])

    assert "本行最宽差异也在噪声内，无法排序" in _row(text, "doc_hit_rate@1")   # 差 0.02 ≤ 0.03
    assert "噪声内" not in _row(text, "doc_mrr")                              # 差 0.10 > 0.03
    assert "噪声地板" in text                                                # 有汇总警告块


def test_render_marks_head_vs_last_only_when_middle_is_far():
    """三列场景：首末很接近、中间很远 → 只标"首末差异在噪声内"（不能说三者无法排序）"""
    a = _report("a", {"doc_ndcg@5": 0.50}, floor=0.03, by_metric={"doc_ndcg@5": 0.03})
    b = _report("b", {"doc_ndcg@5": 0.70}, floor=0.03, by_metric={"doc_ndcg@5": 0.03})
    c = _report("c", {"doc_ndcg@5": 0.51}, floor=0.03, by_metric={"doc_ndcg@5": 0.03})
    text = report.render([a, b, c], ["a", "b", "c"])
    row = _row(text, "doc_ndcg@5")
    assert "首末差异在噪声内" in row
    assert "本行最宽差异也在噪声内" not in row      # 最宽一对是 0.20，明显可分


def test_render_without_noise_floor_marks_nothing():
    a = _report("a", {"doc_hit_rate@1": 0.50})
    b = _report("b", {"doc_hit_rate@1": 0.51})
    text = report.render([a, b], ["a", "b"])
    assert "噪声内" not in text
    assert "噪声地板" not in text


# --------------------------------------------------------------------------- 口径不一致拦截（四项）
def test_caliber_conflicts_detects_all_four_kinds():
    base = _report("base", {"doc_hit_rate@1": 0.5})
    other = _report("other", {"doc_hit_rate@1": 0.5}, top_k=8, candidate_k=32,
                    backend="dashscope", dim=1024, rerank=True)
    warnings = report.caliber_conflicts([base, other])
    joined = " | ".join(warnings)
    assert len(warnings) == 4, warnings       # k / 后端+维度 / candidate_k / 重排
    assert "top_k" in joined
    assert "嵌入口径" in joined
    assert "candidate_k" in joined
    assert "重排" in joined


def test_caliber_conflicts_silent_when_aligned():
    assert report.caliber_conflicts([_report("a", {}), _report("b", {})]) == []


# --------------------------------------------------------------------------- 显示名
def test_metric_label_covers_all_three_calibers():
    assert "文档级" in report.metric_label("doc_hit_rate@1")
    assert "对抗口径" in report.metric_label("kw_precision@3")
    assert "片段级" in report.metric_label("chunk_lift@3")
    assert "lift" in report.metric_label("chunk_lift@3")
