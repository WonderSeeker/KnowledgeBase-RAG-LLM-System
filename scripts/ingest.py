"""命令行入库脚本：把 txt / pdf 批量写入 Chroma 向量库

为什么需要它：
    原项目只能通过 Streamlit 页面上传，无法自动化、无法复现。
    做基线记录、RAGAS 评测、分块参数对比实验都需要「同样的语料、同样的入库过程」，
    所以把入库抽成命令行脚本，绕开浏览器。

用法（在项目根目录执行）：
    # 默认入库 assets/ 下全部 txt（项目自带的 3 份中文示例文本）
    .\\.venv\\Scripts\\python.exe scripts\\ingest.py

    # 指定文件或目录
    .\\.venv\\Scripts\\python.exe scripts\\ingest.py data\\手册.pdf

    # 分块对比实验：每组分块参数入一个独立集合（互不污染）
    .\\.venv\\Scripts\\python.exe scripts\\ingest.py --chunk-size 500 --chunk-overlap 50 --collection rag_c500_o50 --rebuild

    # 只想看不同分块参数会切出多少片段（不调 embedding、不需要 API Key）
    .\\.venv\\Scripts\\python.exe scripts\\ingest.py --dry-run --chunk-size 1000 --chunk-overlap 100

前提：真正入库需要 .env 里的 DASHSCOPE_API_KEY（要用通义 embedding 接口）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 让脚本无论从哪里启动，都能 import 到项目根目录下的模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ★ 显式导入：本脚本会直接 new chromadb.Client（见 drop_collection），
#   必须在建客户端之前完成遥测屏蔽与持久化设置，不能依赖 import 链的巧合
from chroma_quiet import CHROMA_SETTINGS                                # noqa: E402
import config_data as config                                            # noqa: E402
import embeddings as embeddings_module                                  # noqa: E402
from knowledge_base import KnowledgeBaseService, extract_text           # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter     # noqa: E402

SUPPORTED_SUFFIX = {".txt", ".pdf"}


def collect_files(target: Path) -> list[Path]:
    """收集待入库文件：传文件就返回它本身，传目录就递归找 txt/pdf"""
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_SUFFIX else []
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIX)


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """与 KnowledgeBaseService 里用的是同一套分块参数"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=config.separators,
        length_function=len,
    )


def dry_run(files: list[Path], chunk_size: int, chunk_overlap: int) -> int:
    """只做本地切分统计，不写库、不调 embedding：分块对比实验的第一手依据"""
    splitter = build_splitter(chunk_size, chunk_overlap)
    total_chunks = 0
    print(f"[试算] chunk_size={chunk_size} chunk_overlap={chunk_overlap}"
          f"（判定已统一交给 splitter；原文短于 chunk_size 时自然只有 1 个片段）")
    for file_path in files:
        text = extract_text(file_path.read_bytes(), file_path.name)
        pieces = splitter.split_text(text)
        lengths = [len(piece) for piece in pieces]
        total_chunks += len(pieces)
        note = "  ← 原文短于 chunk_size" if len(pieces) == 1 and lengths[0] <= chunk_size else ""
        print(f"  {file_path.name}: 原文 {len(text)} 字 → {len(pieces)} 片段，"
              f"片段长度 min/avg/max = {min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}{note}")
    print(f"[试算] 合计 {total_chunks} 个片段（这是检索的最小单位，也是 BM25 的语料规模）")
    return 0


def drop_collection(name: str) -> int:
    """删除集合，返回原有片段数。

    直接用 chromadb 删，而不是"先构造入库服务再删"——后者会白跑一次嵌入维度探测
    （日志里会多出一遍 [嵌入] 输出，容易让人以为重复入库了）。
    """
    import chromadb
    client = chromadb.Client(CHROMA_SETTINGS)
    try:
        collection = client.get_collection(name)
    except Exception:
        return 0
    count = collection.count()
    client.delete_collection(name)
    return count


def main(argv: list[str] | None = None) -> int:
    """argv 留出参数位：evaluation/experiments.py 与 evaluation/run_ir.py 都以函数方式驱动入库"""
    parser = argparse.ArgumentParser(description="知识库入库（支持 txt / pdf，支持分块参数实验）")
    parser.add_argument("path", nargs="?", default=None, help="文件或目录，默认 assets/")
    parser.add_argument("--chunk-size", type=int, default=None, help="覆盖 config.chunk_size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="覆盖 config.chunk_overlap")
    parser.add_argument("--collection", default=None, help="写入哪个集合，默认按嵌入后端派生（rag / rag_local）")
    parser.add_argument("--embedding-backend", choices=["dashscope", "local"], default=None,
                        help="嵌入后端：dashscope 云端 / local 本地 bge-small-zh-v1.5（默认取 config）")
    parser.add_argument("--rebuild", action="store_true", help="先删除同名集合再重建（做实验时用）")
    parser.add_argument("--dry-run", action="store_true", help="只统计切分结果，不写库、不调 embedding")
    args = parser.parse_args(argv)

    chunk_size = config.chunk_size if args.chunk_size is None else args.chunk_size
    chunk_overlap = config.chunk_overlap if args.chunk_overlap is None else args.chunk_overlap
    backend = args.embedding_backend or config.embedding_backend
    collection = args.collection or embeddings_module.default_collection_name(backend)

    target = Path(args.path).resolve() if args.path else PROJECT_ROOT / "assets"
    if not target.exists():
        print(f"[错误] 路径不存在：{target}")
        return 1

    files = collect_files(target)
    if not files:
        print(f"[跳过] {target} 下没有可入库的 txt/pdf 文件")
        return 0

    if args.dry_run:
        return dry_run(files, chunk_size, chunk_overlap)

    print(f"[入库] 目标：{target}")
    print(f"[入库] 集合={collection} 嵌入后端={backend} "
          f"分块={chunk_size}/{chunk_overlap} 待处理 {len(files)} 个文件")

    if args.rebuild:
        before = drop_collection(collection)        # 先删旧集合，避免新旧分块混在一起
        print(f"[重建] 已删除集合 {collection}（原 {before} 个片段）")
        # md5 记录是按内容去重的，重建集合后必须清掉，否则同内容会被判为"已入库"而跳过
        # 注意：config.md5_path 是 str（阶段 1 改成绝对路径时定的类型），不是 Path
        md5_file = Path(config.md5_path)
        if md5_file.exists():
            md5_file.unlink()
            print("[重建] 已清空 md5.text 去重记录（否则重复内容不会被重新写入）")

    service = KnowledgeBaseService(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        collection_name=collection, embedding_backend=backend,
    )

    success = 0
    for file_path in files:
        start_time = time.perf_counter()
        text = extract_text(file_path.read_bytes(), file_path.name)
        result = service.upload_by_str(text, file_path.name)
        cost = time.perf_counter() - start_time
        print(f"  → {file_path.name}：{result}（含读取，本文件共 {cost:.2f}s）")
        if "Success" in result:
            success += 1

    print(f"\n[入库] 完成：共 {len(files)} 个文件，成功 {success} 个，跳过 {len(files) - success} 个")
    print(f"[入库] 集合 {collection} 现有片段数：{service.chroma._collection.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
