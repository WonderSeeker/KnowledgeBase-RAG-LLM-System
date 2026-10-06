"""检索策略层：向量检索 / BM25 关键词检索 / 混合检索（RRF 融合）/ 重排（Rerank）

阶段 3 的动机：
    基线只有"纯向量 + k=1"。纯向量检索的短板是**字面强匹配类问题**——
    「身高168cm体重100斤」这种精确数值、「gte-rerank-v2」这种专有名词，
    语义上跟语料里的表述并不接近，向量相似度容易把它排到后面，于是漏检。
    BM25 补上字面召回，Rerank 再把粗排结果精排一次，这两层叠加是性价比最高的优化。

四种检索器共用同一个契约：`invoke(question) -> list[Document]`
（与 LangChain 检索器一致，方便直接挂进现有的 LCEL 链）

设计要点：
    1. BM25 的语料直接从 Chroma 集合里读出来，保证**两路检索面对的是同一份语料**；
    2. 混合检索用 RRF（Reciprocal Rank Fusion）而非加权分数相加——
       BM25 与余弦相似度的量纲完全不同，分数直接加权需要调参且不稳定，
       RRF 只看"排名"不看分值，天然免调参，是混合检索的工业默认做法；
    3. 重排失败（网络/额度/模型不可用）时**降级为融合后的顺序并打警告**，不让整条链路挂掉。
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Sequence

import numpy as np
from langchain_core.documents import Document

import config_data as config

# --------------------------------------------------------------------------- 中文分词
_JIEBA = None


def tokenize(text: str) -> list[str]:
    """中文分词：优先用 jieba，装不上时退化为字符二元组

    BM25 的核心是词频统计，中文不分词的 BM25 效果会明显偏弱，所以 jieba 值得这个依赖。
    """
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba
            jieba.setLogLevel(logging.WARNING)      # 别让"Building prefix dict"刷屏
            _JIEBA = jieba
        except ImportError:
            print("[检索] 未安装 jieba，BM25 退化为字符二元组分词（建议装 jieba）")
            _JIEBA = False

    if _JIEBA:
        return [word for word in _JIEBA.lcut(text) if word.strip()]

    compact = "".join(text.split())
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[i:i + 2] for i in range(len(compact) - 1)]


def _doc_key(doc: Document) -> str:
    """用内容哈希做片段指纹，供 RRF 融合时去重（同一片段被两路都召回只算一次）"""
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- 语料加载
def load_all_documents(vector_store) -> list[Document]:
    """把向量库里存着的全部片段读出来，作为 BM25 的语料

    读集合不触发 embedding 调用，所以这个函数本身不需要联网。
    """
    try:
        raw = vector_store._collection.get(include=["documents", "metadatas"])
    except Exception:
        raw = vector_store.get(include=["documents", "metadatas"])

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or [{} for _ in documents]
    return [
        Document(page_content=text, metadata=dict(meta or {}))
        for text, meta in zip(documents, metadatas)
    ]


# --------------------------------------------------------------------------- 各种检索器
class VectorRetriever:
    """向量检索：复用原项目的 Chroma 检索器，只是允许指定返回条数"""

    name = "vector"

    def __init__(self, vector_store_service, k: int):
        self.k = k
        self._retriever = vector_store_service.get_retriever(k=k)

    def invoke(self, question: str) -> list[Document]:
        return list(self._retriever.invoke(question))


class BM25Retriever:
    """BM25 关键词检索：擅长专有名词、型号、精确数值这类字面强匹配的问题"""

    name = "bm25"

    def __init__(self, documents: Sequence[Document], k: int):
        self.k = k
        self.documents = list(documents)
        self._tokenized = [tokenize(doc.page_content) for doc in self.documents]
        if self._tokenized and any(self._tokenized):
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

    def invoke(self, question: str) -> list[Document]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(question))
        order = np.argsort(scores)[::-1][: self.k]

        hits: list[Document] = []
        for index in order:
            if scores[index] <= 0:          # 完全不相干的片段不要塞进来当噪声
                continue
            doc = self.documents[int(index)]
            hits.append(Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": round(float(scores[index]), 4)},
            ))
        return hits


def rrf_fuse(result_lists: Sequence[Sequence[Document]], rrf_k: int = 60, top_n: int = 10) -> list[Document]:
    """RRF 融合：score = Σ 1/(rrf_k + 排名)

    只看排名不看分值，所以不需要为两路检索的量纲差异调权重。
    """
    scores: dict[str, float] = {}
    pool: dict[str, Document] = {}

    for docs in result_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            pool.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    fused: list[Document] = []
    for key, score in ordered[:top_n]:
        doc = pool[key]
        fused.append(Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rrf_score": round(score, 6)},
        ))
    return fused


class DashScopeReranker:
    """重排：用 DashScope 的 gte-rerank-v2 对粗排候选做精排

    重排的价值：向量/BM25 都是"召回导向"的粗排，重排模型是"相关性导向"的精排，
    能把粗排里排在第 5 但其实最相关的片段提到第 1，直接改善 context_precision。
    """

    name = "dashscope-rerank"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from dashscope import TextReRank
        self._TextReRank = TextReRank
        self.model = model or config.rerank_model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.last_error: str | None = None

    def rerank(self, question: str, documents: Sequence[Document], top_n: int) -> list[Document]:
        documents = list(documents)
        if not documents:
            return []

        if not self.api_key:
            self.last_error = "未配置 DASHSCOPE_API_KEY"
            print(f"[检索] 重排已跳过：{self.last_error}（保持粗排顺序）")
            return documents[:top_n]

        try:
            response = self._TextReRank.call(
                model=self.model,
                query=question,
                documents=[doc.page_content for doc in documents],
                top_n=min(top_n, len(documents)),
                return_documents=False,
                api_key=self.api_key,
            )
        except Exception as exc:                      # 网络/额度/参数问题都不该让问答挂掉
            self.last_error = str(exc)
            print(f"[检索] 重排调用失败，降级为粗排顺序：{exc}")
            return documents[:top_n]

        if getattr(response, "status_code", None) != 200:
            self.last_error = f"{getattr(response, 'code', '')} {getattr(response, 'message', '')}".strip()
            print(f"[检索] 重排返回异常（{self.last_error}），降级为粗排顺序")
            return documents[:top_n]

        results = getattr(getattr(response, "output", None), "results", None) or []
        if not results:
            print("[检索] 重排结果为空，降级为粗排顺序")
            return documents[:top_n]

        ordered = sorted(results, key=lambda item: item.get("relevance_score", 0), reverse=True)
        reranked: list[Document] = []
        for item in ordered[:top_n]:
            doc = documents[item["index"]]
            reranked.append(Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(float(item.get("relevance_score", 0)), 4)},
            ))
        return reranked


class HybridRetriever:
    """混合检索：多路召回 → RRF 融合 → 可选重排 → 截断到 k"""

    name = "hybrid"

    def __init__(self, legs: Sequence[object], k: int, rrf_k: int = 60,
                 candidate_k: int = 10, reranker: DashScopeReranker | None = None):
        self.legs = list(legs)
        self.k = k
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.reranker = reranker

    def invoke(self, question: str) -> list[Document]:
        result_lists = [leg.invoke(question) for leg in self.legs]
        fused = rrf_fuse(result_lists, self.rrf_k, self.candidate_k)
        if self.reranker is not None:
            return self.reranker.rerank(question, fused, self.k)
        return fused[: self.k]


# --------------------------------------------------------------------------- 组装
def build_retriever(method: str, vector_leg=None, documents: Sequence[Document] | None = None,
                    k: int | None = None, candidate_k: int | None = None, rrf_k: int | None = None,
                    reranker: DashScopeReranker | None = None):
    """按检索策略组装检索器

    :param method: vector / bm25 / hybrid
    :param vector_leg: 向量检索腿（已按 candidate_k 粗排好候选）
    :param documents: BM25 语料（通常来自 load_all_documents）
    :param reranker: 传入则对融合结果做精排
    """
    k = config.retrieve_k if k is None else k
    candidate_k = config.candidate_k if candidate_k is None else candidate_k
    rrf_k = config.rrf_k if rrf_k is None else rrf_k

    legs: list[object] = []
    if method in ("vector", "hybrid"):
        if vector_leg is None:
            raise ValueError(f"检索策略 {method} 需要 vector_leg（向量检索腿）")
        legs.append(vector_leg)
    if method in ("bm25", "hybrid"):
        legs.append(BM25Retriever(documents or [], k=candidate_k))
    if not legs:
        raise ValueError(f"未知的检索策略：{method}（可选 vector / bm25 / hybrid）")

    # 注意：这里刻意**不做**"单路就直接返回"的短路优化。
    # 各条腿都是按 candidate_k 粗排的（返回条数 > k），必须经过 HybridRetriever
    # 统一截断到 k，否则 vector 策略会返回 candidate_k 个片段，悄悄改变基线行为。
    return HybridRetriever(legs, k=k, rrf_k=rrf_k, candidate_k=candidate_k, reranker=reranker)


def describe(method: str, k: int, candidate_k: int, rerank_enabled: bool) -> str:
    """一句话描述当前检索配置，启动时打出来，便于确认实验条件"""
    parts = [f"策略={method}", f"最终 k={k}"]
    if method == "hybrid":
        parts.append(f"粗排候选={candidate_k}")
    parts.append(f"重排={'开（' + config.rerank_model + '）' if rerank_enabled else '关'}")
    return "，".join(parts)
