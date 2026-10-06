"""对比实验驱动：一次跑多组分块参数 / 检索策略，直接产出横向对比表（阶段 3）

为什么要有它：
    "分块大小/重叠度的对比实验"如果靠手工一组一组跑，很容易漏记参数、把不同口径的数据混在一起。
    这里把「入库 → 评测 → 汇总成一张表」串成一条命令，保证每组实验条件都被记录进报告 meta。

用法：
    # 离线跑通实验流程（不需要 Key；分块在本地真实生效，检索用离线替身）
    .\\.venv\\Scripts\\python.exe evaluation\\experiments.py --mode mock ^
        --configs 300/30,600/60,1000/100 --retriever bm25 --top-k 4

    # 真实实验（需要 Key）：每组分块参数入独立集合，再用 RAGAS 打分
    .\\.venv\\Scripts\\python.exe evaluation\\experiments.py --mode real --name chunk ^
        --configs 500/50,1000/100,1500/200 --retriever hybrid --top-k 4 --candidate-k 10 --rerank

产物：
    evaluation/reports/<name>_c<size>_o<overlap>_<mode>.json   每组一份报告
    evaluation/reports/<name>_<mode>_compare.md                横向对比表
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


def run_one(mode: str, name: str, chunk_size: int, chunk_overlap: int, extra: list[str]) -> Path | None:
    """跑一组实验：real 模式先入库到独立集合，再评测"""
    run_name = f"{name}_c{chunk_size}_o{chunk_overlap}"
    out_path = REPORTS_DIR / f"report_{run_name}_{mode}.json"

    argv = [
        "--mode", mode,
        "--name", run_name,
        "--chunk-size", str(chunk_size),
        "--chunk-overlap", str(chunk_overlap),
        "--out", str(out_path),
        *extra,
    ]

    if mode == "real":
        collection = f"rag_c{chunk_size}_o{chunk_overlap}"
        print(f"\n{'=' * 78}\n[实验] {run_name}：先入库到集合 {collection}\n{'=' * 78}")
        code = ingest_main([
            str(PROJECT_ROOT / "assets"),
            "--chunk-size", str(chunk_size),
            "--chunk-overlap", str(chunk_overlap),
            "--collection", collection,
            "--rebuild",
        ])
        if code != 0:
            print(f"[实验] {run_name} 入库失败，跳过")
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
    parser.add_argument("--name", default="chunk", help="实验名前缀")
    parser.add_argument("--configs", default="500/50,1000/100",
                        help="分块参数列表，逗号分隔，形如 500/50,1000/100")
    parser.add_argument("--retriever", choices=["vector", "bm25", "hybrid"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--rerank", action="store_true", help="开启重排（仅 real 模式生效）")
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    extra: list[str] = []
    if args.retriever:
        extra += ["--retriever", args.retriever]
    if args.top_k is not None:
        extra += ["--top-k", str(args.top_k)]
    if args.candidate_k is not None:
        extra += ["--candidate-k", str(args.candidate_k)]
    if args.rerank:
        extra += ["--rerank"]

    print(f"[实验] 模式={mode_label(args.mode)} 共 {len(configs)} 组分块参数："
          f"{'，'.join(f'{s}/{o}' for s, o in configs)}")
    print(f"[实验] 检索附加参数：{' '.join(extra) if extra else '（用 config 默认值）'}")

    started = time.perf_counter()
    reports: list[Path] = []
    for chunk_size, chunk_overlap in configs:
        path = run_one(args.mode, args.name, chunk_size, chunk_overlap, extra)
        if path:
            reports.append(path)

    if not reports:
        print("[实验] 没有任何成功的实验组，退出")
        return 1

    # 汇总成一张横向对比表
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
