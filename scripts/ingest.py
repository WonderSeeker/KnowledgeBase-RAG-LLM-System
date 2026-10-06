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

import config_data as config                                            # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库入库（支持 txt / pdf，支持分块参数实验）")
    parser.add_argument("path", nargs="?", default=None, help="文件或目录，默认 assets/")
    parser.add_argument("--chunk-size", type=int, default=None, help="覆盖 config.chunk_size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="覆盖 config.chunk_overlap")
    parser.add_argument("--collection", default=None, help="写入哪个集合，默认 config.collection_name")
    parser.add_argument("--rebuild", action="store_true", help="先删除同名集合再重建（做实验时用）")
    parser.add_argument("--dry-run", action="store_true", help="只统计切分结果，不写库、不调 embedding")
    args = parser.parse_args()

    chunk_size = config.chunk_size if args.chunk_size is None else args.chunk_size
    chunk_overlap = config.chunk_overlap if args.chunk_overlap is None else args.chunk_overlap
    collection = args.collection or config.collection_name

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
    print(f"[入库] 集合={collection} 分块={chunk_size}/{chunk_overlap} 待处理 {len(files)} 个文件")

    service = KnowledgeBaseService(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, collection_name=collection
    )

    if args.rebuild:
        before = service.chroma._collection.count()
        service.chroma.delete_collection()          # 删掉旧集合，避免新旧分块混在一起
        print(f"[重建] 已删除集合 {collection}（原 {before} 个片段）")
        service = KnowledgeBaseService(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, collection_name=collection
        )
        # md5 记录是按内容去重的，重建集合后必须清掉，否则同内容会被判为"已入库"而跳过
        if config.md5_path.exists():
            Path(config.md5_path).unlink()
            print("[重建] 已清空 md5.text 去重记录（否则重复内容不会被重新写入）")

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
