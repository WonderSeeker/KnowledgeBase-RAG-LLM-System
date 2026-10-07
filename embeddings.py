"""嵌入后端：DashScope（云端）与本地 OpenAI 兼容服务（127.0.0.1:11434）

为什么单独抽出来、并做「指纹校验」：
    嵌入是整个检索的根。换后端就会换向量维度，而**不同维度的向量放进同一个 collection，
    相似度结果完全是垃圾**——而且不会报错，只会静默给出错误排序。
    所以这里把「后端 + 模型 + 维度」收敛成一个 profile，并在打开 collection 时做三件事：

      1) local 后端先探活：服务没起就给出启动提示，而不是崩；
      2) 用**一次真实嵌入调用**测出实际维度
         —— 不要读 /health 的 dim：模型未加载时它是 null（服务是懒加载的）；
      3) 把指纹写进 collection metadata；若已有指纹不一致，**直接报错退出**。

实测记录（2026-10-07，本机）
    POST http://127.0.0.1:11434/v1/embeddings  →  bge-small-zh-v1.5，维度 512，L2 归一化（范数=1.0）
    GET  http://127.0.0.1:11434/health         →  顶层路径；/v1/health 是 404
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import config_data as config

# 期望维度：仅用于交叉校验，**实际维度始终以真实调用结果为准**
#   dashscope 一列未实测（本机没有可用 Key），故为 None —— 拿不准的数字不写死
EXPECTED_DIMS: dict[str, dict[str, int | None]] = {
    "dashscope": {"text-embedding-v4": None},
    "local": {"bge-small-zh-v1.5": 512},          # 已实测
}

# 集合名后缀：不同后端默认落在不同集合，从命名上就避免混用
COLLECTION_SUFFIX = {"dashscope": "", "local": "_local"}


@dataclass(frozen=True)
class EmbeddingProfile:
    backend: str
    model: str
    base_url: str | None = None
    api_key: str | None = None

    @property
    def expected_dim(self) -> int | None:
        return EXPECTED_DIMS.get(self.backend, {}).get(self.model)

    def fingerprint(self, dim: int) -> dict[str, Any]:
        return {
            "embedding_backend": self.backend,
            "embedding_model": self.model,
            "embedding_dim": int(dim),
        }

    def describe(self) -> str:
        return f"后端={self.backend} 模型={self.model}"


# --------------------------------------------------------------------------- 解析与构造
def resolve_profile(backend: str | None = None) -> EmbeddingProfile:
    backend = backend or config.embedding_backend
    if backend == "dashscope":
        return EmbeddingProfile(backend="dashscope", model=config.embedding_model_name)
    if backend == "local":
        return EmbeddingProfile(
            backend="local",
            model=config.local_embedding_model,
            base_url=config.local_embedding_base_url,
            api_key=config.local_embedding_api_key,
        )
    raise ValueError(f"未知的嵌入后端：{backend}（可选 dashscope / local）")


def default_collection_name(backend: str | None = None) -> str:
    """后端感知的集合名：显式覆盖优先；否则 dashscope -> rag，local -> rag_local

    命名只是"方便隔离"，真正的保证是 collection metadata 里的嵌入指纹校验
    （见 verify_and_stamp）——即使有人手工指定了同名集合，也会被拦下。
    """
    if config.collection_override:
        return config.collection_override
    backend = backend or config.embedding_backend
    if backend not in COLLECTION_SUFFIX:
        raise ValueError(f"未知的嵌入后端：{backend}")
    return f"{config.collection_name}{COLLECTION_SUFFIX[backend]}"


def preflight_local(profile: EmbeddingProfile, timeout: int = 3) -> None:
    """local 后端探活：**顶层 /health**（不是 /v1/health），判据只看 embedding.available

    注意：冷启动时 `loaded=false`、`dim=null` 都是**正常**的（服务刻意懒加载，
    一次探活不该吃掉 200MB/3.1GB 内存），不要据此判定不可用。
    """
    if profile.backend != "local":
        return
    url = f"{profile.base_url}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            info = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"本地嵌入服务未启动或不可达（{url}）：{exc}\n"
            f"启动命令：cd D:\\Projects ; .\\svc.ps1 start llm"
        ) from exc

    embedding_info = info.get("embedding") or {}
    if not embedding_info.get("available"):
        raise RuntimeError(
            f"本地嵌入模型不可用（{url} 返回 embedding.available=false）：{embedding_info}"
        )


def api_base_url(profile: EmbeddingProfile) -> str:
    """OpenAI 客户端用的 base_url —— 必须带 `/v1`

    坑（实测踩到）：本地服务暴露两个不同根路径：
        - 探活：  GET  {base}/health          ← 顶层，**不带** /v1
        - 嵌入：  POST {base}/v1/embeddings   ← OpenAI 兼容路由，**带** /v1
    OpenAI 客户端会在 base_url 之后拼 `/embeddings`，
    所以这里必须补上 `/v1`；写成裸 base 会得到 404（错误信息还很含糊：`{'detail': 'Not Found'}`）。
    """
    base = (profile.base_url or "").rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def build_embeddings(profile: EmbeddingProfile | None = None):
    """按 profile 构造 LangChain 嵌入对象"""
    profile = profile or resolve_profile()
    if profile.backend == "dashscope":
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings(model=profile.model)

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=profile.model,
        base_url=api_base_url(profile),     # ★ 带 /v1，见 api_base_url 的说明
        api_key=profile.api_key,
        # ★ 必须关掉：否则 OpenAIEmbeddings 会先用 tiktoken 做上下文长度预处理，
        #   对中文会报错/算错 token，且完全没必要
        check_embedding_ctx_length=False,
    )


# --------------------------------------------------------------------------- 维度与指纹
_DIM_CACHE: dict[tuple[str, str], int] = {}


def observed_dim(profile: EmbeddingProfile, embeddings) -> int:
    """真实调用一次嵌入，测出**实际**维度（进程内缓存）。

    这是唯一的权威来源；EXPECTED_DIMS 只用来交叉校验，防止"模型名没变但权重换了维度"。
    """
    key = (profile.backend, profile.model)
    if key in _DIM_CACHE:
        return _DIM_CACHE[key]

    vector = embeddings.embed_query("嵌入维度探测")
    dim = len(vector)

    expected = profile.expected_dim
    if expected is not None and dim != expected:
        raise RuntimeError(
            f"嵌入维度与常量表不一致：{profile.describe()} "
            f"实测={dim} 期望={expected}。\n"
            f"请核对 EXPECTED_DIMS（embeddings.py）或确认模型是否被替换。"
        )
    _DIM_CACHE[key] = dim
    return dim


def verify_and_stamp(collection, profile: EmbeddingProfile, dim: int, collection_name: str) -> None:
    """校验 collection 的嵌入指纹；缺失则写入，不一致则报错退出"""
    fingerprint = profile.fingerprint(dim)
    existing = dict(collection.metadata or {})

    conflicts: list[str] = []
    for key, new_value in fingerprint.items():
        old_value = existing.get(key)
        if old_value is None:
            continue                       # 老集合没有该字段：视为未知，不判冲突，稍后补写
        if key == "embedding_dim":
            if int(old_value) != int(new_value):
                conflicts.append(f"  {key}: 集合={old_value}  当前={new_value}")
        elif old_value != new_value:
            conflicts.append(f"  {key}: 集合={old_value}  当前={new_value}")

    if conflicts:
        raise RuntimeError(
            f"集合 {collection_name} 的嵌入指纹与本进程不一致：\n"
            + "\n".join(conflicts)
            + "\n不同嵌入后端/维度的向量**不可混用**（相似度结果没有意义）。\n"
            + f"请换一个集合名重建，例如：\n"
            + f"  .\\.venv\\Scripts\\python.exe scripts\\ingest.py --embedding-backend {profile.backend} "
              f"--collection {collection_name}_{profile.backend} --rebuild"
        )

    if any(existing.get(key) != value for key, value in fingerprint.items()):
        collection.modify(metadata={**existing, **fingerprint})
        print(f"[嵌入] 集合 {collection_name} 写入/补齐嵌入指纹：{fingerprint}")


def prepare(backend: str | None = None, verbose: bool = True):
    """一站式准备：解析 profile → local 探活 → 构造 → 实测维度

    :return: (profile, embeddings, dim)
    """
    profile = resolve_profile(backend)
    preflight_local(profile)
    embeddings = build_embeddings(profile)
    dim = observed_dim(profile, embeddings)
    if verbose:
        print(f"[嵌入] {profile.describe()} 实测维度={dim}")
    return profile, embeddings, dim
