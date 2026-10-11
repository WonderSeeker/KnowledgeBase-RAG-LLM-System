"""CI summary 脚本（scripts/ci_summary.py）

为什么值得给它写测试：CI 页面上那句「本 job 不跑 IR 评测」是**防止误读**的关键 ——
它一旦被改掉或删掉，公开的绿勾就会被当成"所有测试都过了"。
所以这里钉住：统计数字正确 + 声明必须存在且提到 IR 与 Key。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ci_summary  # noqa: E402

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="4" failures="1" errors="0" skipped="1">
    <testcase classname="tests.a" name="test_ok" />
    <testcase classname="tests.a" name="test_ok2" />
    <testcase classname="tests.a" name="test_bad"><failure message="boom" /></testcase>
    <testcase classname="tests.a" name="test_skip"><skipped message="no reason" /></testcase>
  </testsuite>
</testsuites>
"""


def test_summarize_counts_each_outcome(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT, encoding="utf-8")
    text = ci_summary.summarize(path)
    assert "总计 **4** 条" in text
    assert "1 失败" in text and "1 跳过" in text and "2 通过" in text


def test_summarize_lists_skipped_case_names(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT, encoding="utf-8")
    assert "test_skip" in ci_summary.summarize(path)


def test_summarize_declares_what_ci_does_not_cover(tmp_path):
    """★ 核心：绿勾必须自带"我没覆盖什么"的说明"""
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT, encoding="utf-8")
    text = ci_summary.summarize(path)
    assert "不跑 IR 评测" in text
    assert "不跑 RAGAS" in text
    assert "DASHSCOPE_API_KEY" in text
    assert "随机变红" in text


def test_summarize_survives_missing_file(tmp_path):
    """拿不到统计也要出声明，且不能抛异常（CI summary 步骤不该二次失败）"""
    text = ci_summary.summarize(tmp_path / "not_there.xml")
    assert "拿不到统计" in text
    assert "不跑 IR 评测" in text


def test_summarize_survives_broken_xml(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text("<testsuites><oops", encoding="utf-8")
    assert "拿不到统计" in ci_summary.summarize(path)


def test_main_writes_to_github_step_summary(tmp_path, monkeypatch, capsys):
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT, encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert ci_summary.main([str(junit)]) == 0
    assert "不跑 IR 评测" in summary.read_text(encoding="utf-8")
    capsys.readouterr()


def test_main_prints_locally_without_env(tmp_path, monkeypatch, capsys):
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT, encoding="utf-8")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    assert ci_summary.main([str(junit)]) == 0
    assert "没有 GITHUB_STEP_SUMMARY" in capsys.readouterr().out


@pytest.mark.parametrize("expected", ["### 用例统计", "### 本次 CI 的范围（诚实声明）"])
def test_summary_sections(tmp_path, expected):
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT, encoding="utf-8")
    assert expected in ci_summary.summarize(junit)
