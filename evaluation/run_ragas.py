"""跑评测：RAG 链路现场产出答案与检索片段 → RAGAS 打分 → 落 JSON 报告

一条命令跑出报告：
    # 1) 真实 RAGAS 评测（需要 .env 里的 DASHSCOPE_API_KEY）
    .\\.venv\\Scripts\\python.exe evaluation\\run_ragas.py --mode real --name baseline

    # 2) 离线自检（不需要 Key，用启发式代理指标验证流程可跑通）
    .\\.venv\\Scripts\\python.exe evaluation\\run_ragas.py --mode mock

    # 3) 只重新打分，不重跑模型（换指标/换裁判时省钱又省时间）
    .\\.venv\\Scripts\\python.exe evaluation\\run_ragas.py --reuse evaluation\\runs\\baseline.jsonl

4 个指标的含义（对应你要求的 faithfulness / answer_relevancy / context_recall / context_precision）：
    faithfulness      答案是否忠于检索片段（越低说明幻觉越多）
    answer_relevancy  答案是否切题
    context_recall    检索片段是否覆盖了标准答案（越低说明漏检）
    context_precision 检索片段里有用的比例（越低说明噪声多）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config_data as config                                    # noqa: E402
from dataset import EvalItem, load_eval_set, summarize          # noqa: E402

RUNS_DIR = EVAL_DIR / "runs"
REPORTS_DIR = EVAL_DIR / "reports"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


# --------------------------------------------------------------------------- 工具
def clean_json(obj):
    """把 NaN / Infinity 换成 None，否则 json.dump 会写出非法 JSON"""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 4)
    return obj


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_json(data), f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- 第一步：跑 RAG 链路
def run_pipeline(items: list[EvalItem], mode: str, top_k: int) -> list[dict]:
    """对每条问题跑一次 RAG，收集 answer / contexts / 耗时

    返回的 records 会先存成 jsonl，之后重新打分可以直接 --reuse，不必再跑模型。
    """
    config.retrieve_k = top_k          # 必须在建 RagService 之前覆盖，检索器是在 __init__ 里建的

    if mode == "mock":
        from offline import lexical_retrieve, load_local_chunks, mock_answer
        chunks = load_local_chunks()
        print(f"[离线模式] 本地切分出 {len(chunks)} 个片段，用字面检索代替向量检索（不联网）")
        records = []
        for item in items:
            start = time.perf_counter()
            hits = lexical_retrieve(item.question, chunks, top_k)
            contexts = [hit["content"] for hit in hits]
            answer = mock_answer(item.question, contexts)
            records.append({
                "id": item.id,
                "question": item.question,
                "ground_truth": item.ground_truth,
                "source": item.source,
                "tags": item.tags,
                "answer": answer,
                "contexts": contexts,
                "n_contexts": len(contexts),
                "total_seconds": round(time.perf_counter() - start, 3),
            })
            print(f"  · {item.id} 命中 {len(contexts)} 段")
        return records

    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit(
            "未找到 DASHSCOPE_API_KEY。\n"
            f"请把 Key 填进 {PROJECT_ROOT / '.env'}，或改用 --mode mock 先跑通流程。"
        )

    from rag import RagService
    rag = RagService()
    records = []
    for i, item in enumerate(items, start=1):
        result = rag.answer(item.question, session_id=f"eval_{item.id}")
        records.append({
            "id": item.id,
            "question": item.question,
            "ground_truth": item.ground_truth,
            "source": item.source,
            "tags": item.tags,
            "answer": result["answer"],
            # 检索片段落盘时加一点长度保护，避免巨大的片段把 jsonl 撑爆
            "contexts": [ctx[:2000] for ctx in result["contexts"]],
            "n_contexts": result["n_contexts"],
            "retrieve_seconds": result["retrieve_seconds"],
            "generate_seconds": result["generate_seconds"],
            "total_seconds": result["total_seconds"],
        })
        print(f"  · [{i}/{len(items)}] {item.id} 命中 {result['n_contexts']} 段，"
              f"耗时 {result['total_seconds']}s")
    return records


def load_records(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- 第二步：打分
def score_with_ragas(records: list[dict], judge_provider: str, judge_model: str) -> tuple[dict, list[dict]]:
    """接入 RAGAS 打分，返回（汇总指标, 每题明细）"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from langchain_community.embeddings import DashScopeEmbeddings

    judge_llm = build_judge_llm(judge_provider, judge_model)
    judge_embeddings = LangchainEmbeddingsWrapper(
        DashScopeEmbeddings(model=config.embedding_model_name)
    )

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=judge_embeddings,
        raise_exceptions=False,        # 单条打分失败不要中断整轮评测
        show_progress=True,
    )

    frame = result.to_pandas()
    per_question: list[dict] = []
    for record, (_, row) in zip(records, frame.iterrows()):
        per_question.append({**record, **{name: row.get(name) for name in METRIC_NAMES}})

    metrics = {name: (None if frame[name].isna().all() else float(frame[name].mean()))
               for name in METRIC_NAMES if name in frame.columns}
    return metrics, per_question


def build_judge_llm(provider: str, model: str):
    """裁判模型。默认走 DashScope 的 OpenAI 兼容端点——这是 RAGAS 最稳的接法"""
    if provider == "dashscope":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
        )
    if provider == "ollama":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key="ollama",                       # 本地服务不校验密钥，占位即可
            base_url="http://127.0.0.1:11434/v1",
            temperature=0,
        )
    if provider == "dashscope-native":
        from langchain_community.chat_models.tongyi import ChatTongyi
        return ChatTongyi(model=model)
    raise SystemExit(f"未知的裁判来源：{provider}（可选 dashscope / ollama / dashscope-native）")


def score_with_heuristic(records: list[dict]) -> tuple[dict, list[dict]]:
    """离线启发式打分，指标口径与 RAGAS 同名但算法不同（见 offline.py 的说明）"""
    from offline import heuristic_scores

    per_question: list[dict] = []
    for record in records:
        scores = heuristic_scores(
            record["question"], record["answer"], record["contexts"], record["ground_truth"]
        )
        per_question.append({**record, **scores})

    metrics = {
        name: sum(r[name] for r in per_question) / len(per_question)
        for name in METRIC_NAMES
    }
    return metrics, per_question


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 评测")
    parser.add_argument("--mode", choices=["real", "mock"], default="real",
                        help="real=真模型+RAGAS；mock=离线自检（不联网、不需要 Key）")
    parser.add_argument("--name", default="baseline", help="本轮实验名，用于文件命名（如 baseline / optimized）")
    parser.add_argument("--out", default=None, help="报告输出路径，默认 evaluation/report_<name>.json")
    parser.add_argument("--eval-set", default=str(EVAL_DIR / "eval_set.jsonl"), help="评测集路径")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 config.retrieve_k（阶段 3 做 A/B 用）")
    parser.add_argument("--judge", choices=["ragas", "heuristic"], default=None,
                        help="裁判：ragas=真 RAGAS；heuristic=离线代理指标。real 模式默认 ragas")
    parser.add_argument("--judge-provider", default="dashscope",
                        choices=["dashscope", "ollama", "dashscope-native"], help="裁判模型来源")
    parser.add_argument("--judge-model", default="qwen-plus", help="裁判模型名（打分用，别用太慢的模型）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，调试用")
    parser.add_argument("--reuse", default=None, help="复用已有的 runs jsonl，跳过 RAG 链路只重新打分")
    args = parser.parse_args()

    judge = args.judge or ("heuristic" if args.mode == "mock" else "ragas")
    top_k = args.top_k if args.top_k is not None else config.retrieve_k
    if args.out:
        out_path = Path(args.out)
    elif args.mode == "mock":
        out_path = REPORTS_DIR / f"report_{args.name}_mock.json"
    else:
        out_path = EVAL_DIR / f"report_{args.name}.json"

    started = time.perf_counter()
    print("=" * 78)
    print(f"[评测] 实验名={args.name} 模式={args.mode} 裁判={judge} 检索 top_k={top_k}")
    print("=" * 78)

    # 1) 拿答案与检索片段
    if args.reuse:
        reuse_path = Path(args.reuse)
        records = load_records(reuse_path)
        print(f"[评测] 复用已有结果：{reuse_path}（{len(records)} 条），跳过 RAG 链路")
    else:
        items = load_eval_set(Path(args.eval_set))
        if args.limit:
            items = items[:args.limit]
        print(summarize(items))
        print("-" * 78)
        records = run_pipeline(items, args.mode, top_k)
        run_path = RUNS_DIR / f"{args.name}_{args.mode}.jsonl"
        save_records(run_path, records)
        print(f"[评测] 运行明细已保存：{run_path}")

    # 2) 打分
    print("-" * 78)
    if judge == "ragas":
        if args.mode == "mock" or args.reuse:
            if not os.environ.get("DASHSCOPE_API_KEY"):
                raise SystemExit("RAGAS 打分需要 DASHSCOPE_API_KEY（裁判模型）。请填入 .env，或改用 --judge heuristic")
        metrics, per_question = score_with_ragas(records, args.judge_provider, args.judge_model)
    else:
        metrics, per_question = score_with_heuristic(records)

    # 3) 出报告
    elapsed = time.perf_counter() - started
    total_seconds = [r.get("total_seconds") or 0 for r in records]
    report = {
        "meta": {
            "name": args.name,
            "mode": args.mode,
            "judge": judge,
            "judge_provider": args.judge_provider if judge == "ragas" else None,
            "judge_model": args.judge_model if judge == "ragas" else None,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_seconds": round(elapsed, 2),
            "eval_set": str(Path(args.eval_set)),
            "eval_size": len(records),
            "retrieval": {
                "method": "本地字面检索（离线替身）" if args.mode == "mock" else "向量检索（Chroma）",
                "top_k": top_k,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "max_spliter_char_number": config.max_spliter_char_number,
            },
            "models": {
                "embedding": config.embedding_model_name,
                "chat": config.chat_model_name,
            },
            "versions": {
                "python": sys.version.split()[0],
                "ragas": _version_of("ragas"),
                "langchain": _version_of("langchain"),
            },
            "avg_total_seconds": round(sum(total_seconds) / len(records), 3) if records else None,
            "avg_n_contexts": round(sum(r["n_contexts"] for r in records) / len(records), 2) if records else None,
            "warning": ("MOCK 数据：检索与打分都是离线替身，仅用于验证评测流程可跑通，不代表真实效果"
                        if args.mode == "mock" or judge == "heuristic" else None),
        },
        "metrics": metrics,
        "per_question": per_question,
    }
    write_json(out_path, report)

    print("=" * 78)
    for name in METRIC_NAMES:
        value = metrics.get(name)
        print(f"  {name:<18} {'N/A' if value is None else f'{value:.4f}'}")
    print("=" * 78)
    print(f"[评测] 报告已写入：{out_path}")
    if report["meta"]["warning"]:
        # 注意：Windows 控制台默认 GBK，打印 emoji 会 UnicodeEncodeError，这里只用纯文本
        print(f"[警告] {report['meta']['warning']}")
    return 0


def _version_of(package: str) -> str:
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
