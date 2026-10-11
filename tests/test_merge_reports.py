"""重复测量的合并逻辑（evaluation/run_ir.py 的 merge_reports）

这段代码把 N 个独立进程的结果合成一份报告，并算出**该配置自己的噪声地板**。
它承载着两个容易犯的错，都在这里钉死：

  1) ★ **尺度混用**：不能拿"所有指标极差的最大值"当统一地板 ——
     `lift` 的量纲是 4~8、`hit_rate` 是 0~1，混在一起算会得出"11.5 题 / 6"这种荒唐值。
     所以 headline 只取主指标（PRIMARY_NOISE_METRICS）里的最大值。
  2) 逐指标极差必须完整保留（对比表要按行取用它）。
"""

from __future__ import annotations

import json

import pytest

import run_ir


def _child(name: str, metrics: dict, eval_size: int = 36) -> dict:
    return {
        "meta": {
            "name": name,
            "mode": "ir",
            "judge": "deterministic",
            "eval_size": eval_size,
            "generated_at": "2026-10-11 00:00:00",
            "warning": "单次运行未测噪声地板",
            "retrieval": {"method": "vector", "top_k": 4, "candidate_k": 10},
        },
        "metrics": metrics,
        "per_question": [{"id": "q-1"}],
    }


@pytest.fixture()
def children(tmp_path, monkeypatch):
    """造 3 份"子进程报告"，并把 REPORTS_DIR 指到临时目录"""
    monkeypatch.setattr(run_ir, "REPORTS_DIR", tmp_path)
    metrics = [
        # doc_hit_rate@1：极差 0.05（3 题 / 36 量级）
        # kw_lift@3    ：极差 2.0  ← 量纲完全不同，绝不能主导 headline
        {"doc_hit_rate@1": 0.50, "doc_mrr": 0.60, "kw_lift@3": 5.0},
        {"doc_hit_rate@1": 0.55, "doc_mrr": 0.65, "kw_lift@3": 7.0},
        {"doc_hit_rate@1": 0.45, "doc_mrr": 0.55, "kw_lift@3": 6.0},
    ]
    paths = []
    for index, item in enumerate(metrics, start=1):
        path = tmp_path / f"report_r{index}_ir.json"
        path.write_text(json.dumps(_child(f"r{index}", item), ensure_ascii=False), encoding="utf-8")
        paths.append(path)
    return paths


def test_metrics_are_averaged(children, tmp_path):
    merged = json.loads(run_ir.merge_reports(children, "merged").read_text(encoding="utf-8"))
    assert merged["metrics"]["doc_hit_rate@1"] == pytest.approx(0.5)      # (0.5+0.55+0.45)/3
    assert merged["metrics"]["doc_mrr"] == pytest.approx(0.6)
    assert merged["metrics"]["kw_lift@3"] == pytest.approx(6.0)


def test_per_metric_spread_is_kept(children):
    merged = json.loads(run_ir.merge_reports(children, "merged").read_text(encoding="utf-8"))
    spreads = merged["meta"]["noise_floor"]["by_metric"]
    assert spreads["doc_hit_rate@1"] == pytest.approx(0.10)     # 0.55 - 0.45
    assert spreads["kw_lift@3"] == pytest.approx(2.0)           # 7.0 - 5.0


def test_headline_floor_excludes_large_scale_metrics(children):
    """★ 回归：headline 不能被 lift 这类"大量纲"指标带跑偏"""
    merged = json.loads(run_ir.merge_reports(children, "merged").read_text(encoding="utf-8"))
    floor = merged["meta"]["noise_floor"]
    assert floor["value"] == pytest.approx(0.10), "headline 应当只取主指标的极差"
    assert floor["max_by_metric"] == pytest.approx(2.0), "最大极差另存一处，供人查看"
    assert "kw_lift@3" not in floor["headline_metrics"]
    assert floor["items"] == pytest.approx(3.6)                 # 0.10 * 36 题


def test_meta_records_repeat_provenance(children):
    merged = json.loads(run_ir.merge_reports(children, "merged").read_text(encoding="utf-8"))
    meta = merged["meta"]
    assert meta["repeat_n"] == 3
    assert meta["repeat_runs"] == ["r1", "r2", "r3"]
    assert "均值" in meta["warning"]                 # 覆盖掉子进程那句"单次运行"
    assert "per_question" in meta["per_question_note"]


def test_merged_report_is_written_to_reports_dir(children, tmp_path):
    out = run_ir.merge_reports(children, "merged")
    assert out.parent == tmp_path
    assert out.name == "report_merged_ir.json"
    assert out.exists()


def test_single_child_has_zero_spread(tmp_path, monkeypatch):
    """只跑 1 次时极差恒为 0（--repeat 1 的退化情形，不能除零/报错）"""
    monkeypatch.setattr(run_ir, "REPORTS_DIR", tmp_path)
    path = tmp_path / "report_only_ir.json"
    path.write_text(json.dumps(_child("only", {"doc_hit_rate@1": 0.5}), ensure_ascii=False),
                    encoding="utf-8")
    merged = json.loads(run_ir.merge_reports([path], "merged_one").read_text(encoding="utf-8"))
    assert merged["meta"]["repeat_n"] == 1
    assert merged["meta"]["noise_floor"]["value"] == 0.0
