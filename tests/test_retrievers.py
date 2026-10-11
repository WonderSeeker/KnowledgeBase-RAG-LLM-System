"""检索层（retrievers.py）：分词、RRF 融合、BM25、向量腿排序、重排降级、组装

两个重点：

1) ★ **单路检索必须仍然截断到 k**
   `build_retriever` 曾经对"单路且不重排"做短路优化，直接返回那条腿 ——
   而各条腿是按 `candidate_k` 粗排的（返回条数 > k），
   于是"纯向量 + k=1"会返回 candidate_k 条，**静默改变基线行为**（不报错，基线数据全废）。
   现在一律过 `HybridRetriever` 统一截断，这里钉死。

2) ★ **排序必须与输入顺序无关**
   向量腿改为显式按 `(距离, 片段指纹)` 排序，RRF 按 `(-分数, 指纹)` ——
   并列时不再看"谁先返回"。这样跨进程的顺序抖动被消掉（剩下的差异是候选集合级的，
   那部分消不掉，只能靠 `--repeat` 测量并披露，见 CHANGES §4.11.2）。
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

import retrievers
from retrievers import (
    BM25Retriever,
    DashScopeReranker,
    VectorRetriever,
    _doc_key,
    build_retriever,
    rrf_fuse,
    tokenize,
)


def _doc(text: str, **metadata) -> Document:
    return Document(page_content=text, metadata=dict(metadata))


class _FakeVectorStore:
    """只实现 VectorRetriever 用到的那一个方法"""

    def __init__(self, pairs):
        self._pairs = pairs

    def similarity_search_with_score(self, question, k):      # noqa: ARG002
        return list(self._pairs)


class _FakeVectorService:
    def __init__(self, pairs):
        self.vector_store = _FakeVectorStore(pairs)


class _FakeLeg:
    """一条假的检索腿：永远返回固定的一批文档"""

    name = "fake"

    def __init__(self, docs):
        self.docs = list(docs)

    def invoke(self, question):                               # noqa: ARG002
        return list(self.docs)


# --------------------------------------------------------------------------- 分词与指纹
def test_tokenize_returns_non_empty_tokens_for_chinese():
    tokens = tokenize("身高168cm体重100斤穿什么码")
    assert tokens and all(token.strip() for token in tokens)
    assert tokenize("身高168cm") == tokenize("身高168cm")      # 确定性


def test_doc_key_depends_only_on_content():
    assert _doc_key(_doc("同样内容", source="a")) == _doc_key(_doc("同样内容", source="b"))
    assert _doc_key(_doc("内容甲")) != _doc_key(_doc("内容乙"))


# --------------------------------------------------------------------------- RRF 融合
def test_rrf_fuse_is_order_independent_on_ties():
    """两路排名互为逆序时，两篇文档的 RRF 分数**完全相等** → 结果只能由指纹决定"""
    first, second = _doc("甲", source="a"), _doc("乙", source="b")
    forward = rrf_fuse([[first, second], [second, first]], rrf_k=60, top_n=2)
    backward = rrf_fuse([[second, first], [first, second]], rrf_k=60, top_n=2)
    assert [doc.page_content for doc in forward] == [doc.page_content for doc in backward]
    # 并列时按指纹升序 —— 也就是"顺序由内容决定，与谁先返回无关"
    expected = [doc.page_content for doc in
                sorted([first, second], key=lambda doc: _doc_key(doc))]
    assert [doc.page_content for doc in forward] == expected


def test_rrf_fuse_dedupes_same_chunk_across_legs():
    """同一片段被两路都召回只算一次（否则它会因"两路都靠前"被重复计分）"""
    same = _doc("同一片段", source="a")
    fused = rrf_fuse([[same], [same]], rrf_k=60, top_n=5)
    assert len(fused) == 1
    assert "rrf_score" in fused[0].metadata


def test_rrf_fuse_respects_top_n():
    docs = [_doc(f"文档{i}") for i in range(5)]
    assert len(rrf_fuse([docs], rrf_k=60, top_n=2)) == 2


# --------------------------------------------------------------------------- 向量腿排序
def test_vector_retriever_sorts_by_distance_then_fingerprint():
    near, far = _doc("近的", source="a"), _doc("远的", source="b")
    # Chroma 返回 (文档, 距离)，距离越小越近；这里故意打乱顺序
    service = _FakeVectorService([(far, 0.9), (near, 0.1)])
    hits = VectorRetriever(service, k=2).invoke("问题")
    assert [doc.page_content for doc in hits] == ["近的", "远的"]


def test_vector_retriever_is_stable_for_equal_distances():
    """★ 距离完全相等时，顺序必须由指纹决定 —— 与返回顺序无关"""
    first, second = _doc("甲"), _doc("乙")
    forward = VectorRetriever(_FakeVectorService([(first, 0.5), (second, 0.5)]), k=2).invoke("q")
    backward = VectorRetriever(_FakeVectorService([(second, 0.5), (first, 0.5)]), k=2).invoke("q")
    assert [doc.page_content for doc in forward] == [doc.page_content for doc in backward]


# --------------------------------------------------------------------------- BM25
# ★ 注意：BM25Okapi 的 IDF 在小语料 / 高频词上会 ≤ 0，而实现会把**非正分**的片段过滤掉
#   （"完全不相干的片段不要塞进来当噪声"）。所以这里的语料必须够大、目标词必须够稀有，
#   否则测出来是空列表 —— 那是 BM25 的正常行为，不是 bug。
def _corpus_with(target: str, hits: int = 3, fillers: int = 9) -> list[Document]:
    docs = [_doc(f"{target}相关的第{i}段说明", source=f"hit{i}.txt") for i in range(hits)]
    docs += [_doc(f"完全无关的填充内容第{i}段，讲的是另外一件事", source=f"fill{i}.txt")
             for i in range(fillers)]
    return docs


def test_bm25_hits_on_literal_terms():
    docs = _corpus_with("身高168cm建议M码")
    hits = BM25Retriever(docs, k=3).invoke("身高168cm")
    assert hits, "至少应命中一条"
    assert all("身高168cm" in doc.page_content for doc in hits)
    assert "bm25_score" in hits[0].metadata


def test_bm25_drops_zero_score_documents():
    """完全不相干的片段不要塞进来当噪声"""
    docs = _corpus_with("羽绒服怎么洗")
    assert BM25Retriever(docs, k=3).invoke("量子纠缠") == []


def test_bm25_respects_k():
    docs = _corpus_with("羽绒服清洗方法", hits=3, fillers=9)
    assert len(BM25Retriever(docs, k=2).invoke("羽绒服清洗")) == 2


# --------------------------------------------------------------------------- 组装（★ 截断回归）
def test_single_leg_retriever_still_truncates_to_k():
    """★ 回归：单路检索也必须截断到 k（否则 k=1 会返回 candidate_k 条，静默改变基线）"""
    leg = _FakeLeg([_doc(f"片段{i}") for i in range(10)])
    retriever = build_retriever("vector", vector_leg=leg, documents=None, k=1, candidate_k=10)
    assert len(retriever.invoke("问题")) == 1


def test_hybrid_retriever_returns_k_documents_with_rrf_score():
    leg = _FakeLeg([_doc("共享片段"), _doc("向量独有片段")])
    docs = [_doc("共享片段"), _doc("BM25 独有片段")]
    retriever = build_retriever("hybrid", vector_leg=leg, documents=docs, k=2, candidate_k=4)
    hits = retriever.invoke("问题")
    assert len(hits) == 2
    assert all("rrf_score" in doc.metadata for doc in hits)


def test_build_retriever_rejects_unknown_method():
    with pytest.raises(ValueError):
        build_retriever("nope", vector_leg=_FakeLeg([]), documents=[], k=1, candidate_k=1)


def test_vector_method_requires_vector_leg():
    with pytest.raises(ValueError):
        build_retriever("vector", vector_leg=None, documents=[], k=1, candidate_k=1)


# --------------------------------------------------------------------------- 重排降级
def test_reranker_degrades_without_api_key(no_api_key):
    """★ 重排依赖外部 API：没有 Key / 网络抖动时必须降级返回粗排顺序，而不是抛异常"""
    reranker = DashScopeReranker(api_key=None)
    docs = [_doc("甲"), _doc("乙"), _doc("丙")]
    result = reranker.rerank("问题", docs, top_n=2)
    assert len(result) == 2
    assert reranker.last_error is not None


def test_describe_mentions_rerank_state():
    text = retrievers.describe("hybrid", k=4, candidate_k=10, rerank_enabled=True)
    assert "hybrid" in text
