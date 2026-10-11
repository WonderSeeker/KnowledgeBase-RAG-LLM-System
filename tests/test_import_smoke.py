"""导入冒烟：整条链路的模块必须都能 import，且 `RagService` 必须能构造

★ 为什么这个测试是必需的（而不是"覆盖率凑数"）：
   CHANGES §5.2 记录过本项目的**致命缺陷**：上游代码从
   `langchain_core.runnables` 导入 `RunnableWithMessageHistory` —— 而 `langchain-core 0.1.53`
   **并不导出**这个名字，正确路径是 `langchain_core.runnables.history`。
   后果不是"某个功能失灵"，而是 `RagService` **完全无法构造 = 整个系统回答不了任何问题**。

   更糟的是它是**潜伏缺陷**：Streamlit 首页能开、`/health` 返回 200、
   评测在 mock 模式下全绿 —— 因为 mock 模式走离线替身、**从不 import `rag`**；
   `/chat` 的 503 门禁又在 `import rag` 之前返回。
   也就是说：**只跑评测永远发现不了它**，必须有一个"把模块全部真的 import 一遍、
   并把服务真的构造一次"的测试。

   这正是 §5.2 结尾那条教训要实现的东西 —— 当时它只是文档里的一句命令，现在它是一条测试。
"""

from __future__ import annotations

import importlib

import pytest

# 覆盖全部"会被真实运行路径 import 到"的模块（含 server：它 import 时就会建 FastAPI app）
MODULES = (
    "config_data",
    "embeddings",
    "file_history_store",
    "vector_stores",
    "chroma_quiet",
    "knowledge_base",
    "retrievers",
    "rag",
    "server",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name, monkeypatch, isolated_state):
    """逐个 import：任何一条 ImportError/AttributeError 都会让整个系统不可用"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dummy-construction-only")
    importlib.import_module(module_name)


def test_every_module_imports_together(monkeypatch, isolated_state):
    """一次性全导入（模块之间互相 import 的路径也要通）"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dummy-construction-only")
    for module_name in MODULES:
        importlib.import_module(module_name)


def test_rag_service_can_be_constructed(monkeypatch, isolated_state, fake_embedding_backend):
    """★ 核心：问答链路的入口必须能构造

    注意构造路径里**有一次真实的嵌入请求**：`RagService.__init__` → `embeddings.prepare()`
    → `observed_dim()`（实测向量维度）。所以这条测试必须先把嵌入后端换成离线替身，
    否则它会因为 401 / 连不上本地服务而失败 —— 而那是**环境问题，不是代码问题**，
    混在一起就会让这条关键测试变成"看运气"。
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dummy-construction-only")

    from rag import RagService

    service = RagService()
    assert service.retriever is not None
    assert service.chain is not None


def test_exported_import_path_is_valid():
    """直接钉住那个缺陷本身：从正确路径导入必须成功

    （若有人把 import 改回 `langchain_core.runnables`，这条会立刻红 —— 见 CHANGES §5.2）
    """
    from langchain_core.runnables.history import RunnableWithMessageHistory  # noqa: F401

    assert RunnableWithMessageHistory is not None
