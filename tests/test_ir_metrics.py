"""IR 指标（evaluation/ir_metrics.py）：纯函数，确定性，不需要任何外部依赖

这些指标是项目"效果结论"的地基，所以逐个钉死语义 ——
特别是 `precision@k` 的分母、`lift` 的换算，以及 `base_rate`（跨分块可比性的关键）。
"""

from __future__ import annotations

import pytest

import ir_metrics


# --------------------------------------------------------------------------- 单点指标
def test_hit_rate_counts_at_least_one_hit():
    ranking = ["a", "b", "c"]
    assert ir_metrics.hit_rate_at_k(ranking, {"c"}, 1) == 0.0     # top-1 里没有
    assert ir_metrics.hit_rate_at_k(ranking, {"c"}, 3) == 1.0     # top-3 里有


def test_hit_rate_without_positive_is_zero():
    assert ir_metrics.hit_rate_at_k(["a"], set(), 5) == 0.0


def test_recall_is_coverage_of_positives():
    assert ir_metrics.recall_at_k(["a", "b", "x"], {"a", "b"}, 2) == 1.0
    assert ir_metrics.recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5


def test_precision_denominator_is_always_k():
    """★ 关键语义：分母固定是 k，不是"实际返回条数"

    这条决定了 precision 能不能跨分块配置比较（见 CHANGES 4.10.2：
    基准率漂移时 raw precision 会被基准率主导，所以必须配 lift 一起看）。
    """
    assert ir_metrics.precision_at_k(["a", "b"], {"a"}, 2) == 0.5
    # 只返回了 1 条、但 k=4 → 分母仍是 4（"没召回"要算成噪声，而不是被忽略）
    assert ir_metrics.precision_at_k(["a"], {"a"}, 4) == 0.25
    assert ir_metrics.precision_at_k(["a"], {"a"}, 0) == 0.0


def test_reciprocal_rank_uses_first_hit_position():
    assert ir_metrics.reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert ir_metrics.reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert ir_metrics.reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_is_one_for_perfect_ranking():
    assert ir_metrics.ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(1.0)
    # 正例排得越靠后，nDCG 越低
    assert ir_metrics.ndcg_at_k(["x", "a"], {"a"}, 2) < 1.0
    assert ir_metrics.ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_ndcg_without_positive_is_zero():
    assert ir_metrics.ndcg_at_k(["a"], set(), 3) == 0.0


# --------------------------------------------------------------------------- 排序去重
def test_dedupe_keep_order_keeps_first_occurrence():
    """同一份文档命中多次时，只算它第一次出现的位置

    不去重会让 hit/mrr/nDCG 虚高 —— 文档级评测里"同一份文档命中两次"没有意义。
    """
    assert ir_metrics.dedupe_keep_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert ir_metrics.dedupe_keep_order(["a", "", "b"]) == ["a", "b"]     # 空 id 丢掉


# --------------------------------------------------------------------------- 批量 + 前缀
def test_evaluate_rankings_averages_and_prefixes():
    rankings = [["a", "b"], ["x", "y"]]
    relevants = [{"a"}, {"x"}]
    metrics = ir_metrics.evaluate_rankings(rankings, relevants, k_values=(1,), prefix="doc")
    assert metrics["doc_hit_rate@1"] == 1.0    # 两题都在 top-1 命中
    assert metrics["doc_recall@1"] == 1.0      # 每题各 1 个正例，都被覆盖
    assert all(name.startswith("doc_") for name in metrics)


def test_evaluate_rankings_rejects_length_mismatch():
    with pytest.raises(ValueError):
        ir_metrics.evaluate_rankings([["a"]], [], k_values=(1,))


def test_evaluate_rankings_empty_input():
    assert ir_metrics.evaluate_rankings([], [], k_values=(1,)) == {}


# --------------------------------------------------------------------------- 基准率与 lift
def test_base_rate_is_positive_share_of_corpus():
    """基准率 = 平均正例数 / 语料片段数 —— 也就是"随机取一个片段恰好相关"的概率"""
    relevants = [{"a", "b"}, {"a", "b"}]        # 平均 2 个正例
    assert ir_metrics.base_rate(relevants, 20) == pytest.approx(0.1)
    assert ir_metrics.base_rate([set()], 20) == 0.0
    assert ir_metrics.base_rate([{"a"}], 0) == 0.0      # 语料为空时不能除零


def test_add_lift_normalizes_precision_by_base_rate():
    """lift@k = precision@k / 基准率（解决的正是"precision 下界随基准率漂移"）"""
    metrics = {"chunk_precision@1": 0.30, "chunk_precision@3": 0.20}
    out = ir_metrics.add_lift(dict(metrics), "chunk", base=0.10, k_values=(1, 3))
    assert out["chunk_lift@1"] == pytest.approx(3.0)
    assert out["chunk_lift@3"] == pytest.approx(2.0)


def test_add_lift_is_noop_without_base_rate():
    """基准率为 0（没有正例）时不能除零，原样返回"""
    metrics = {"chunk_precision@1": 0.3}
    assert ir_metrics.add_lift(dict(metrics), "chunk", base=0.0, k_values=(1,)) == metrics


def test_add_lift_skips_missing_precision_keys():
    out = ir_metrics.add_lift({}, "chunk", base=0.5, k_values=(1,))
    assert out == {}


# --------------------------------------------------------------------------- 指标名与口径文本
def test_metric_names_include_lift_and_prefix():
    names = ir_metrics.metric_names(k_values=(1, 3), prefix="kw")
    assert "kw_lift@1" in names and "kw_lift@3" in names
    assert "kw_ndcg@3" in names and "kw_mrr" in names


def test_caliber_text_marks_circularity_and_adversarial_intent():
    """口径文本必须自带"这个口径偏袒谁、有没有循环性"，否则数字会被误读"""
    doc = ir_metrics.caliber_text("doc")
    kw = ir_metrics.caliber_text("kw")
    chunk = ir_metrics.caliber_text("chunk")
    assert "文档级" in doc
    assert "对抗性口径" in kw and "偏袒" in kw          # 明说它偏袒关键词检索
    assert "弱标注" in chunk and "同源" in chunk         # 明说它循环
