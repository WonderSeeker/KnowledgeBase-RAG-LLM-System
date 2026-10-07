"""FastAPI 包装：把 RAG 问答能力暴露成 HTTP 接口（阶段 4）

启动方式（二选一）：
    cd D:\\Projects\\KnowledgeBase-RAG-LLM-System
    .\\.venv\\Scripts\\python.exe server.py            # 默认 127.0.0.1:8020
    cd D:\\Projects; .\\svc.ps1 start kbapi              # 同上，日志进 D:\\Projects\\_logs\\kbapi.out

接口：
    GET  /health   健康检查：状态 + Key 是否配置 + 语料规模 + 当前检索配置（**不需要 Key**）
    GET  /config   当前检索/分块/模型配置，用于确认实验条件
    POST /chat     问答：{question, session_id} -> 答案 + 命中片段 + 分段耗时

设计要点：
    1. RagService 懒加载 —— 服务能先起来（/health 随时可用），不会因为缺 Key 就启动失败；
    2. 缺 Key / 语料为空 返回 503 + 可操作的提示，而不是 500 堆栈；
    3. 上游（DashScope）调用异常返回 502，并保留原始错误信息，便于定位；
    4. 输出一律不用 emoji：Windows 控制台是 GBK，emoji 会让 print 抛 UnicodeEncodeError。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException                     # noqa: E402
from pydantic import BaseModel, Field                          # noqa: E402

# ★ 显式导入：本模块会自己 new chromadb.Client（见 _chroma_count），
#   必须在建客户端之前完成遥测 logger 屏蔽与持久化设置，不能依赖 import 链的巧合
from chroma_quiet import CHROMA_SETTINGS                       # noqa: E402
import config_data as config                                   # noqa: E402
import embeddings as embeddings_module                         # noqa: E402

API_VERSION = "0.1.0"

app = FastAPI(
    title="KnowledgeBase RAG API",
    version=API_VERSION,
    description="本地知识库 RAG 问答接口（检索策略：vector / bm25 / hybrid + 可选重排）",
)

# --------------------------------------------------------------------------- 懒加载与工具
_rag: Any = None
_rag_lock = threading.Lock()
_chroma_client: Any = None


def _api_key_present() -> bool:
    return bool(os.environ.get("DASHSCOPE_API_KEY"))


def _chroma_count() -> int | None:
    """读向量库里的片段数。任何异常都吞掉并返回 None——/health 必须永远能返回"""
    global _chroma_client
    try:
        if _chroma_client is None:
            import chromadb
            # 复用 vector_stores 里的客户端设置：同时也会触发 chromadb 遥测 logger 的屏蔽
            # （直接 new 一个 Settings 会绕过那边的处理，/health 又会被 telemetry 报错刷屏）
            from vector_stores import CHROMA_SETTINGS
            _chroma_client = chromadb.Client(CHROMA_SETTINGS)
        return _chroma_client.get_or_create_collection(
            embeddings_module.default_collection_name()
        ).count()
    except Exception as exc:                      # noqa: BLE001
        print(f"[api] 读取向量库片段数失败（不影响 /health）：{exc}")
        return None


def _get_rag():
    """首次请求时才真正构造 RagService（加载检索器与模型客户端）"""
    global _rag
    if _rag is None:
        with _rag_lock:
            if _rag is None:
                from rag import RagService
                print("[api] 首次请求，正在初始化 RAG 服务...")
                _rag = RagService()
    return _rag


def _retrieval_config() -> dict:
    return {
        "method": config.retrieval_method,
        "top_k": config.retrieve_k,
        "candidate_k": config.candidate_k,
        "rrf_k": config.rrf_k if config.retrieval_method == "hybrid" else None,
        "rerank": config.rerank_enabled,
        "rerank_model": config.rerank_model if config.rerank_enabled else None,
        "collection": embeddings_module.default_collection_name(),
    }


def _chunking_config() -> dict:
    return {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }


def _models_config() -> dict:
    """⚠️ expected_dim 是**配置声明的期望值**，不是实测值。

    真实维度要发起一次嵌入调用才知道（本地服务冷启动时 /health 的 dim 是 null），
    所以这里不敢冒充实测结果；实际使用中的维度以检索时打印的 `[嵌入] … 实测维度=N` 为准。
    """
    profile = embeddings_module.resolve_profile()
    return {
        "embedding": profile.model,
        "chat": config.chat_model_name,
        "embedding_backend": profile.backend,
        "embedding_expected_dim": profile.expected_dim,
    }


def _not_ready(error: str, message: str, hint: str) -> HTTPException:
    """统一的服务不可用错误体：让调用方一眼看出该做什么，而不是去翻堆栈"""
    return HTTPException(status_code=503, detail={"error": error, "message": message, "hint": hint})


# --------------------------------------------------------------------------- 数据模型
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str = Field("api_user", max_length=64, description="会话 id，用于多轮历史（写入 chat_history/）")


class ChatResponse(BaseModel):
    answer: str
    contexts: list[str] = Field(..., description="命中的检索片段原文，便于核对答案依据")
    n_contexts: int = Field(..., description="命中片段数，等于实际生效的 top_k（可能因无相关内容而更少）")
    session_id: str
    retrieve_seconds: float | None = None
    generate_seconds: float | None = None
    total_seconds: float | None = None
    retrieval: dict[str, Any] = Field(..., description="本次回答实际使用的检索配置")


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok=可正常问答；degraded=进程活着但还不能问答")
    reasons: list[str] = Field(default_factory=list, description="degraded 的原因（缺 Key / 语料为空）")
    api_key_configured: bool
    collection: str
    documents: int | None = Field(None, description="向量库里的片段数；None 表示读取失败")
    retrieval: dict[str, Any]
    chunking: dict[str, Any]
    models: dict[str, Any]
    version: str


# --------------------------------------------------------------------------- 接口
@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "KnowledgeBase RAG API",
        "version": API_VERSION,
        "endpoints": ["GET /health", "GET /config", "POST /chat"],
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查。刻意不依赖 DashScope：缺 Key 时也能反映进程与语料状态"""
    reasons: list[str] = []
    if not _api_key_present():
        reasons.append("未配置 DASHSCOPE_API_KEY")
    documents = _chroma_count()
    if documents == 0:
        reasons.append("向量库为空，尚未入库任何语料")

    return HealthResponse(
        status="ok" if not reasons else "degraded",
        reasons=reasons,
        api_key_configured=_api_key_present(),
        collection=embeddings_module.default_collection_name(),
        documents=documents,
        retrieval=_retrieval_config(),
        chunking=_chunking_config(),
        models=_models_config(),
        version=API_VERSION,
    )


@app.get("/config")
def get_config() -> dict:
    """暴露当前配置，方便确认"这次测的到底是哪套参数" """
    return {
        "retrieval": _retrieval_config(),
        "chunking": _chunking_config(),
        "models": _models_config(),
        "paths": {
            "persist_directory": config.persist_directory,
            "md5_path": config.md5_path,
        },
        "version": API_VERSION,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """检索增强问答。缺 Key / 语料为空 -> 503；上游调用失败 -> 502"""
    if not _api_key_present():
        if config.embedding_backend == "local":
            message = ("未配置 DASHSCOPE_API_KEY：嵌入已切到本地服务，"
                       "但**生成模型仍走通义千问**，所以问答依然需要 Key。")
        else:
            message = "未配置 DASHSCOPE_API_KEY，无法调用通义千问的嵌入与生成接口。"
        raise _not_ready(
            "api_key_missing",
            message,
            f"把 Key 填进 {PROJECT_ROOT / '.env'}，然后 .\\svc.ps1 restart kbapi",
        )
    if _chroma_count() == 0:
        raise _not_ready(
            "knowledge_base_empty",
            "向量库为空，还没有任何语料。",
            "先入库：.\\.venv\\Scripts\\python.exe scripts\\ingest.py",
        )

    started = time.perf_counter()
    print(f"[api] POST /chat session={request.session_id} question={request.question[:60]}")
    try:
        result = _get_rag().answer(request.question, session_id=request.session_id)
    except Exception as exc:                     # noqa: BLE001
        print(f"[api] 问答失败：{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail={
            "error": "upstream_error",
            "message": f"调用模型服务失败：{type(exc).__name__}: {exc}",
            "hint": "检查 DASHSCOPE_API_KEY 是否有效、网络是否可达，或查看 _logs\\kbapi.out",
        }) from exc

    elapsed = time.perf_counter() - started
    print(f"[api] 完成：命中 {result['n_contexts']} 段，总耗时 {elapsed:.2f}s")
    return ChatResponse(
        answer=result["answer"],
        contexts=result["contexts"],
        n_contexts=result["n_contexts"],
        session_id=request.session_id,
        retrieve_seconds=result["retrieve_seconds"],
        generate_seconds=result["generate_seconds"],
        total_seconds=result["total_seconds"],
        retrieval=_retrieval_config(),
    )


# --------------------------------------------------------------------------- 启动
def main() -> int:
    parser = argparse.ArgumentParser(description="KnowledgeBase RAG API 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args, _ = parser.parse_known_args()          # 忽略多余参数，避免和 svc.ps1 的调用方式冲突

    print(f"[api] 启动 KnowledgeBase RAG API v{API_VERSION}")
    print(f"[api] 监听 http://{args.host}:{args.port}  文档 http://{args.host}:{args.port}/docs")
    print(f"[api] 该服务被占用前请确认端口未被其他项目使用（本项目约定 8020）")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
