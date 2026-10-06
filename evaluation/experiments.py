"""对比实验驱动：一次跑多组实验（分块参数 × 检索策略），直接产出横向对比表（阶段 3）

为什么要有它：
    对比实验如果靠手工一组一组跑，很容易漏记参数、把不同口径的数据混在一起
    （最典型的就是「k 不一样还比 precision」这种无效对比）。
    这里把「入库 → 评测 → 汇总成一张表」串成一条命令，并把实验条件全部写进报告 meta。

用法：
    # 离线跑通实验流程（不需要 Key；分块在本地真实生效，检索用离线替身）
    .\\.venv\\Scripts\\python.exe evaluation\\experiments.py --mode mock ^
        --configs 600/60 --retrievers vector,bm25,hybrid --top-k 4 --candidate-k 12

    # 分块参数对比（固定检索策略，保证只有分块这一个变量）
    .\\.venv\\Scripts\\python.exe evaluation\\experiments.py --mode real --name chunk ^
        --configs 500/50,1000/100,1500/200 --retriever hybrid --top-k 4 --candidate-k 10 --rerank

    # 检索策略对比（固定 k，否则 context_precision 天然不可比）
    .\\.venv\\Scripts\\python.exe evaluation\\experiments.py --mode real --name retriever ^
        --configs 1000/100 --retrievers vector,bm25,hybrid --top-k 4 --candidate-k 10

产物：
    evaluation/reports/report_<name>_<...>_<mode>.json   每组一份报告
    evaluation/reports/<name>_<mode>_compare.md          横向对比表
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _path in (str(EVAL_DIR), str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_ragas                                                  # noqa: E402
from ingest import main as ingest_main                            # noqa: E402
from report import load_report, render                            # noqa: E402

REPORTS_DIR = EVAL_DIR / "reports"
MAX_GRID = 12          # 网格上限：超过就提醒，避免一次跑掉半天和一堆 API 额度


def parse_configs(raw: str) -> list[tuple[int, int]]:
    """解析 "300/30,600/60" 这样的分块参数列表"""
    configs: list[tuple[int, int]] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "/" not in piece:
            raise SystemExit(f"分块参数格式错误：{piece}（应为 chunk_size/chunk_overlap，如 600/60）")
        size, overlap = piece.split("/", 1)
        configs.append((int(size), int(overlap)))
    if not configs:
        raise SystemExit("至少要给一组 --configs")
    return configs


def parse_methods(retrievers: str | None, single: str | None) -> list[str | None]:
    """检索策略维度：[None] 表示用 config 里的默认值"""
    if retrievers:
        methods = [m.strip() for m in retrievers.split(",") if m.strip()]
        unknown = [m for m in methods if m not in ("vector", "bm25", "hybrid")]
        if unknown:
            raise SystemExit(f"未知的检索策略：{unknown}（可选 vector / bm25 / hybrid）")
        return methods
    return [single]


def ingest_collection(chunk_size: int, chunk_overlap: int) -> str:
    """按分块参数入库到独立集合（同分块参数只入库一次，多个检索策略共用）"""
    collection = f"rag_c{chunk_size}_o{chunk_overlap}"
    code = ingest_main([
        str(PROJECT_ROOT / "assets"),
        "--chunk-size", str(chunk_size),
        "--chunk-overlap", str(chunk_overlap),
        "--collection", collection,
        "--rebuild",
    ])
    if code != 0:
        raise RuntimeError(f"集合 {collection} 入库失败")
    return collection


def run_one(mode: str, name: str, chunk_size: int, chunk_overlap: int,
            method: str | None, extra: list[str], ingested: set[tuple[int, int]]) -> Path | None:
    """跑一组实验：real 模式先按分块参数入库，再评测"""
    suffix = f"_c{chunk_size}_o{chunk_overlap}" + (f"_{method}" if method else "")
    run_name = f"{name}{suffix}"
    out_path = REPORTS_DIR / f"report_{run_name}_{mode}.json"

    argv = [
        "--mode", mode,
        "--name", run_name,
        "--chunk-size", str(chunk_size),
        "--chunk-overlap", str(chunk_overlap),
        "--out", str(out_path),
        *extra,
    ]
    if method:
        argv += ["--retriever", method]

    if mode == "real":
        print(f"\n{'=' * 78}\n[实验] {run_name}：准备集合\n{'=' * 78}")
        try:
            if (chunk_size, chunk_overlap) not in ingested:
                collection = ingest_collection(chunk_size, chunk_overlap)
                ingested.add((chunk_size, chunk_overlap))
            else:
                collection = f"rag_c{chunk_size}_o{chunk_overlap}"
                print(f"[实验] 集合 {collection} 本轮已入库过，直接复用")
        except RuntimeError as exc:
            print(f"[实验] {exc}，跳过 {run_name}")
            return None
        argv += ["--collection", collection]

    print(f"\n{'=' * 78}\n[实验] {run_name}：开始评测\n{'=' * 78}")
    try:
        run_ragas.main(argv)
    except SystemExit as exc:                  # 缺 Key 等致命错误，直接终止整个实验
        if exc.code not in (0, None):
            raise
    return out_path if out_path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="分块参数 / 检索策略对比实验")
    parser.add_argument("--mode", choices=["real", "mock"], default="real")
    parser.add_argument("--name", default="expr", help="实验名前缀")
    parser.add_argument("--configs", default="1000/100",
                        help="分块参数列表，逗号分隔，形如 500/50,1000/100")
    parser.add_argument("--retrievers", default=None,
                        help="检索策略列表（逗号分隔），如 vector,bm25,hybrid；与 --configs 组成网格")
    parser.add_argument("--retriever", choices=["vector", "bm25", "hybrid"], default=None,
                        help="只跑单一检索策略（等价于 --retrievers 写一个值）")
    parser.add_argument("--top-k", type=int, default=None,
                        help="交给大模型的片段数。⚠️ 对比 context_precision 时必须固定同一个 k")
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--rerank", action="store_true", help="开启重排（仅 real 模式生效）")
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    methods = parse_methods(args.retrievers, args.retriever)

    extra: list[str] = []
    if args.top_k is not None:
        extra += ["--top-k", str(args.top_k)]
    if args.candidate_k is not None:
        extra += ["--candidate-k", str(args.candidate_k)]
    if args.rerank:
        extra += ["--rerank"]

    print(f"[实验] 模式={mode_label(args.mode)}")
    print(f"[实验] 分块维度：{'，'.join(f'{s}/{o}' for s, o in configs)}")
    print(f"[实验] 检索维度：{'，'.join(m or '（config 默认）' for m in methods)}")
    print(f"[实验] 附加参数：{' '.join(extra) if extra else '（用 config 默认值）'}")
    print(f"[实验] 共 {len(configs) * len(methods)} 组")
    if len(configs) * len(methods) > MAX_GRID:
        print(f"[预警] 超过 {MAX_GRID} 组，real 模式会消耗大量 API 额度与时间，请确认后重跑")
    if len(methods) > 1 and args.top_k is None:
        print("[预警] 对比多种检索策略但没显式指定 --top-k："
              "若 config.retrieve_k 变化，context_precision 将不可比")

    started = time.perf_counter()
    reports: list[Path] = []
    ingested: set[tuple[int, int]] = set()
    for chunk_size, chunk_overlap in configs:
        for method in methods:
            path = run_one(args.mode, args.name, chunk_size, chunk_overlap, method, extra, ingested)
            if path:
                reports.append(path)

    if not reports:
        print("[实验] 没有任何成功的实验组，退出")
        return 1

    loaded = [load_report(str(p)) for p in reports]
    labels = [r.get("meta", {}).get("name", p.stem) for r, p in zip(loaded, reports)]
    text = render(loaded, labels, worst=0)

    compare_path = REPORTS_DIR / f"{args.name}_{args.mode}_compare.md"
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    compare_path.write_text(text, encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(text)
    print(f"[实验] 共 {len(reports)} 组，总耗时 {time.perf_counter() - started:.1f}s")
    print(f"[实验] 对比表已写入：{compare_path}")
    return 0


def mode_label(mode: str) -> str:
    return "真实 RAGAS（需要 Key）" if mode == "real" else "离线自检（不联网）"


if __name__ == "__main__":
    raise SystemExit(main())
