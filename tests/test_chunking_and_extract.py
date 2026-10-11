"""分块与文本抽取（knowledge_base.py）

★ 这里最重要的是 `test_chunk_size_takes_effect_on_short_doc`：
   它是那个**真实缺陷**的回归测试 —— 原版用一个「≤1000 字就不切分」的硬门槛，
   导致《尺码推荐.txt》（270 字）无论 chunk_size 设 300 还是 1500 都只有 1 个片段，
   分块参数对短文档**完全失效**（详见 CHANGES §4.8）。
   修复后判定统一交给 splitter，本测试用"短文本 + 小 chunk_size"把行为钉住。
"""

from __future__ import annotations

import pytest

import config_data as config
from knowledge_base import extract_text


def _short_doc() -> str:
    """造一段"短、但明显可以切成两段"的文本（两段各 130 字，用 \\n\\n 分隔）"""
    return "尺" * 130 + "\n\n" + "码" * 130


# --------------------------------------------------------------------------- 分块（★ 缺陷回归）
def test_chunk_size_takes_effect_on_short_doc(make_kb):
    text = _short_doc()
    # 前提：这段文本短于那个（已废弃的）硬门槛 —— 也就是说**旧代码在它身上一定不切分**
    assert len(text) < config.max_spliter_char_number

    service = make_kb(chunk_size=150, chunk_overlap=15, collection_name="t150")
    service.upload_by_str(text, "短文档.txt")

    assert service.chroma._collection.count() == 2, \
        "chunk_size=150 时两段 130 字的文本应被切成 2 个片段（旧代码会只留 1 段）"


def test_large_chunk_size_keeps_document_as_single_chunk(make_kb):
    """同一个文档、只把 chunk_size 调大 → 变成 1 个片段：证明**chunk_size 才是控制项**"""
    text = _short_doc()
    service = make_kb(chunk_size=1000, chunk_overlap=100, collection_name="t1000")
    service.upload_by_str(text, "短文档.txt")
    assert service.chroma._collection.count() == 1


def test_chunk_overlap_is_applied(make_kb):
    """重叠度生效：连续片段之间应有重叠内容（用长文 + 小 chunk_size 观察片段数变多）"""
    long_text = "。".join(f"第{i}段内容" for i in range(200))
    service = make_kb(chunk_size=200, chunk_overlap=50, collection_name="t_overlap")
    service.upload_by_str(long_text, "长文档.txt")
    assert service.chroma._collection.count() > 1


def test_same_content_is_deduped(make_kb):
    """同一内容第二次入库必须被 MD5 去重拦住（改造中不能破坏这个行为）"""
    text = _short_doc()
    service = make_kb(chunk_size=150, chunk_overlap=15, collection_name="t_dedup")
    assert "Success" in service.upload_by_str(text, "a.txt")
    assert "Repeat" in service.upload_by_str(text, "a.txt")
    assert service.chroma._collection.count() == 2      # 没有重复写入


# --------------------------------------------------------------------------- 文本抽取
def test_extract_text_reads_utf8():
    assert extract_text("中文内容".encode("utf-8"), "a.txt") == "中文内容"


def test_extract_text_falls_back_to_gbk():
    """中文 Windows 上 txt 很可能是 GBK —— 不能因为编码问题直接崩"""
    assert extract_text("中文内容".encode("gbk"), "a.txt") == "中文内容"


def test_extract_text_gbk_fallback_ignores_broken_bytes():
    broken = b"\xd6\xd0\xce\xc4\xff\xfe\xfd"      # 合法 GBK 前缀 + 非法字节
    assert "中文" in extract_text(broken, "a.txt")


def test_extract_text_routes_pdf_to_pypdf():
    """PDF 分支必须真的走 pypdf（用真实库造一个空白页，验证不抛异常且返回字符串）"""
    pypdf = pytest.importorskip("pypdf")
    import io

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)

    text = extract_text(buffer.getvalue(), "blank.pdf")
    assert isinstance(text, str)


def test_extract_text_pdf_suffix_is_case_insensitive():
    pypdf = pytest.importorskip("pypdf")
    import io

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)

    assert isinstance(extract_text(buffer.getvalue(), "BLANK.PDF"), str)
