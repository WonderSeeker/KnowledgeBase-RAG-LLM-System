"""pytest 公共装置

★ 本目录的测试全部是**确定性**的：不联网、不需要 DASHSCOPE_API_KEY、不需要本地嵌入服务。
  需要外部资源的东西一律不放进来。

★ 为什么这里没有「IR 评测」的测试（重要，别加回来）：
    IR 指标本身是确定性的，但**向量腿不是** —— chromadb 走 HNSW 近似检索，
    索引图在每个进程里重建，实测**同一配置跨进程会有 24/36 题的候选集合不同**
    （见 CHANGES §4.10.5 与 §4.11.2），`doc_hit@1` 会抖动约 3 题 / 36。
    把它做成 CI 门禁会**随机变红**，所以 IR 评测刻意不进 CI —— 它只在需要时人工跑，
    并且必须带 `--repeat 3` 输出实测噪声地板。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# 三个路径都要加：项目根（import config_data / rag）、evaluation（import dataset / report）、
# scripts（import ingest）
for _path in (ROOT, ROOT / "evaluation", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture()
def project_root() -> Path:
    return ROOT


@pytest.fixture()
def isolated_state(monkeypatch, tmp_path: Path) -> Path:
    """把 config 里所有"会落盘"的路径指到临时目录

    否则跑一次测试就会往真实的 chroma_db/ 与 md5.text 里写东西 —— 测试绝不能污染被测产物。
    """
    import config_data as config

    monkeypatch.setattr(config, "persist_directory", str(tmp_path / "chroma_db"), raising=False)
    monkeypatch.setattr(config, "md5_path", str(tmp_path / "md5.text"), raising=False)
    return tmp_path


@pytest.fixture()
def fake_embedding_backend(monkeypatch):
    """把嵌入后端换成完全离线的轻量替身，返回那个假嵌入对象

    ★ 为什么必须替换，而不是"等真服务起来"：
      `embeddings.prepare()` 内部会调用 `observed_dim()` —— 也就是**真的发一次嵌入请求**
      来实测向量维度（`RagService.__init__` 走的也是这条路，见 rag.py）。
      在没有 Key / 没有本地服务时它会抛 401 或连接错误。
      被测的是**业务逻辑**（分块、检索组装、链路构造），不是嵌入模型本身。
    """
    from langchain_community.embeddings import FakeEmbeddings

    import embeddings as embeddings_module

    class _FakeProfile:
        """真 profile 是个 dataclass；这里只需要后端名（日志里会打出来）"""
        backend = "test"

    fake = FakeEmbeddings(size=64)
    monkeypatch.setattr(
        embeddings_module, "prepare",
        lambda backend=None, verbose=True: (_FakeProfile(), fake, 64),
    )
    monkeypatch.setattr(embeddings_module, "default_collection_name", lambda backend=None: "test_collection")
    monkeypatch.setattr(embeddings_module, "verify_and_stamp", lambda *args, **kwargs: None)
    return fake


@pytest.fixture()
def make_kb(fake_embedding_backend, isolated_state: Path):
    """返回一个"造 KnowledgeBaseService"的工厂（嵌入后端已换成离线替身）"""
    import knowledge_base as kb

    def _make(**kwargs):
        return kb.KnowledgeBaseService(**kwargs)

    return _make


@pytest.fixture()
def no_api_key(monkeypatch):
    """确保测试环境里没有 DASHSCOPE_API_KEY（否则某些代码路径会真的去联网）"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
