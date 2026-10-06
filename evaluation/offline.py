"""零依赖离线评测组件：不需要 API Key，就能把「切分 → 检索 → 生成 → 打分」整条链路跑通

存在的意义：
    1. 没有 DASHSCOPE_API_KEY 时验证评测代码与报告格式是否正确（不烧 token、不联网）；
    2. 提供一个可复现的对照物：检索和打分全在本地，所以能证明
       「检索参数一变，指标就跟着变」这条因果链是通的。

⚠️ 郑重说明：本模块的打分**不是 RAGAS**，而是「字符二元组重叠」的启发式代理指标。
   真实指标必须用 run_ragas.py 的 --mode real（需要 API Key）。
   报告里会写死 "judge": "heuristic"，不要把两者的数字放在一起比较。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config_data as config                                          # noqa: E402
from knowledge_base import extract_text                               # noqa: E402
from langchain_core.documents import Document                         # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter   # noqa: E402

ASSETS_DIR = PROJECT_ROOT / "assets"


# --------------------------------------------------------------------------- 文本相似度
def char_bigrams(text: str) -> set[str]:
    """取字符二元组（bigram）作为中文的轻量"词袋"

    中文没有天然空格分词，用 bigram 既能避免引入 jieba 依赖，又比单字更有区分度。
    """
    compact = "".join(text.split())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def coverage(part: str, whole: str) -> float:
    """part 的内容有多少比例出现在 whole 里（召回视角，取值 0~1）"""
    part_grams = char_bigrams(part)
    if not part_grams:
        return 0.0
    return len(part_grams & char_bigrams(whole)) / len(part_grams)


# --------------------------------------------------------------------------- 本地语料与检索
def load_local_documents() -> list[Document]:
    """用与知识库完全相同的分块参数，在本地把 assets 语料切好（不联网、不调 embedding）"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
        length_function=len,
    )
    documents: list[Document] = []
    for path in sorted(ASSETS_DIR.glob("*.txt")):
        text = extract_text(path.read_bytes(), path.name)
        if len(text) > config.max_spliter_char_number:
            pieces = splitter.split_text(text)
        else:
            pieces = [text]                    # 与 upload_by_str 的判定逻辑保持一致
        for index, piece in enumerate(pieces):
            documents.append(Document(page_content=piece, metadata={"source": path.name, "index": index}))
    return documents


class LocalLexicalRetriever:
    """本地字面检索器：扮演"向量检索腿"的离线替身

    它和 BM25Retriever 的接口完全一致（invoke -> list[Document]），
    这样离线模式就能复用**真实的** RRF 融合、重排适配等阶段 3 代码，
    而不是另写一套假流程——假流程跑通没有意义。
    """

    name = "local-lexical"

    def __init__(self, documents: list[Document], k: int):
        self.documents = list(documents)
        self.k = k

    def invoke(self, question: str) -> list[Document]:
        scored = [(coverage(question, doc.page_content), doc) for doc in self.documents]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            Document(page_content=doc.page_content,
                     metadata={**doc.metadata, "lexical_score": round(score, 4)})
            for score, doc in scored[: self.k]
            if score > 0
        ]


def mock_answer(question: str, contexts: list[str]) -> str:
    """模拟"模型基于检索片段作答"：直接引用最相关片段的前 180 字

    faithfulness 会因此偏高，这是预期的——真实场景下模型会自己组织语言，可能引入幻觉。
    """
    if not contexts:
        return "无相关参考资料"
    return contexts[0][:180]


# --------------------------------------------------------------------------- 启发式代理指标
def heuristic_scores(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    """4 个代理指标，对应 RAGAS 的同名指标，含义对齐、算法不同

    - faithfulness      答案有多少内容能在检索片段里找到（防幻觉）
    - answer_relevancy  答案覆盖了问题多少内容（防答非所问）
    - context_recall    标准答案有多少内容被检索片段覆盖（防漏检）
    - context_precision 检索片段有多少与标准答案相关（防噪声）
    """
    merged_context = "\n".join(contexts)

    if contexts:
        precision = sum(coverage(ground_truth, ctx) for ctx in contexts) / len(contexts)
    else:
        precision = 0.0

    return {
        "faithfulness": round(coverage(answer, merged_context), 4),
        "answer_relevancy": round(coverage(question, answer), 4),
        "context_recall": round(coverage(ground_truth, merged_context), 4),
        "context_precision": round(precision, 4),
    }
