"""评测集（evaluation/dataset.py）：序列化往返、正例回退规则、真实评测集的结构校验

`answer_keywords` 是"对抗性口径"的输入（判定片段是否含答案关键词），
所以它必须真的存在于数据里 —— 早期它只存在于出题脚本、不在 JSONL 里，是这一层发现的。
"""

from __future__ import annotations

import pytest

import dataset
from dataset import EvalItem, load_eval_set, save_eval_set, summarize


def test_eval_item_round_trip_keeps_all_fields():
    item = EvalItem(
        id="q-1", question="问题", ground_truth="答案", source="a.txt",
        tags=["精确数值"], relevant_sources=["a.txt", "b.txt"], answer_keywords=["关键词"],
    )
    again = EvalItem.from_dict(item.to_dict())
    assert again == item
    assert again.answer_keywords == ["关键词"]


def test_eval_item_from_dict_tolerates_missing_optional_fields():
    item = EvalItem.from_dict({"id": "q", "question": "问", "ground_truth": "答"})
    assert item.source == ""
    assert item.tags == []
    assert item.relevant_sources == []
    assert item.answer_keywords == []


def test_relevant_docs_prefers_relevant_sources_then_falls_back_to_source():
    """文档级口径的正例来源必须确定、可复现 —— 这里把回退规则钉死"""
    multi = EvalItem(id="a", question="q", ground_truth="g", source="a.txt",
                     relevant_sources=["a.txt", "b.txt"])
    assert multi.relevant_docs == ["a.txt", "b.txt"]

    single = EvalItem(id="b", question="q", ground_truth="g", source="a.txt")
    assert single.relevant_docs == ["a.txt"]

    empty = EvalItem(id="c", question="q", ground_truth="g", source="")
    assert empty.relevant_docs == []


def test_save_and_load_round_trip(tmp_path):
    items = [EvalItem(id=f"q-{i}", question=f"问{i}", ground_truth=f"答{i}", source="a.txt")
             for i in range(3)]
    path = tmp_path / "eval.jsonl"
    save_eval_set(items, path)
    assert load_eval_set(path) == items


def test_load_eval_set_skips_blank_lines(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text('{"id":"q","question":"问","ground_truth":"答"}\n\n\n', encoding="utf-8")
    assert len(load_eval_set(path)) == 1


# --------------------------------------------------------------------------- 真实评测集
@pytest.fixture()
def committed_eval_set():
    path = dataset.EVAL_DIR / "eval_set_v2.jsonl"
    if not path.exists():
        pytest.skip("eval_set_v2.jsonl 不存在（本仓库应当提交了它）")
    return load_eval_set(path)


def test_committed_eval_set_is_well_formed(committed_eval_set):
    items = committed_eval_set
    assert len(items) >= 20, "评测集规模太小，指标没有统计意义"

    ids = [item.id for item in items]
    assert len(ids) == len(set(ids)), "评测集里有重复 id"

    for item in items:
        assert item.question.strip(), f"{item.id} 问题为空"
        assert item.ground_truth.strip(), f"{item.id} 标准答案为空"
        assert item.relevant_docs, f"{item.id} 没有正例文档（relevant_docs 为空）"
        assert item.tags, f"{item.id} 没有类型标签"


def test_committed_eval_set_carries_answer_keywords(committed_eval_set):
    """★ 对抗性口径依赖它：缺失会让 kw_* 指标全为 0（而且是静默的）"""
    missing = [item.id for item in committed_eval_set if not item.answer_keywords]
    assert not missing, f"这些样本缺 answer_keywords：{missing}"


def test_committed_eval_set_answer_keywords_appear_in_their_source_document(committed_eval_set):
    """每个样本至少要有一个关键词能在**来源文档**里找到

    ★ 校验对象是来源文档，不是标准答案：对抗性口径的标注规则是
    「片段文本含 answer_keywords 即算相关」，正例来自**语料**。
    若某个样本的关键词在语料里一个都找不到，它的 kw_* 指标会**静默全为 0**，
    整行看起来"检索很差"，其实是数据问题。
    """
    corpus_dir = dataset.PROJECT_ROOT / "corpus"
    if not corpus_dir.exists():
        pytest.skip("找不到 corpus/ 目录（语料可能未被提交）")

    missing = []
    for item in committed_eval_set:
        text = ""
        for source in item.relevant_docs:
            path = corpus_dir / source
            if path.exists():
                text += path.read_text(encoding="utf-8")
        if not text:
            continue        # 该样本的语料不在 corpus/ 里（例如 assets 体系）→ 跳过
        if not any(keyword in text for keyword in item.answer_keywords):
            missing.append(item.id)

    assert not missing, f"这些样本的关键词在来源文档里一个都找不到（对抗口径会静默全 0）：{missing}"


def test_summarize_mentions_size(committed_eval_set):
    text = summarize(committed_eval_set)
    assert str(len(committed_eval_set)) in text
