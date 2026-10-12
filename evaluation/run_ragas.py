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
# 两个路径都要加：PROJECT_ROOT 用来 import config_data / rag，EVAL_DIR 用来 import dataset / offline
for _path in (str(EVAL_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config_data as config                                    # noqa: E402
import embeddings as embeddings_module                          # noqa: E402
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
def build_offline_retriever(candidate_k: int):
    """离线模式的检索器：向量腿换成"本地字面检索替身"，其余（BM25/RRF/截断）都是真实代码

    这样离线跑通才真的有意义——验证的是阶段 3 的检索逻辑本身，而不是另写一套假流程。
    """
    from offline import LocalLexicalRetriever, load_local_documents
    from retrievers import build_retriever

    documents = load_local_documents()
    method = config.retrieval_method
    vector_leg = None
    if method in ("vector", "hybrid"):
        # 与真实路径保持同样的粗排口径：向量腿按 candidate_k 出候选，再由 HybridRetriever 截断到 k
        vector_leg = LocalLexicalRetriever(documents, k=candidate_k)
    print(f"[离线模式] 本地切分 {len(documents)} 个片段；向量腿=字面检索替身，策略={method}（不联网）")
    return build_retriever(
        method,
        vector_leg=vector_leg,
        documents=documents,
        k=config.retrieve_k,
        candidate_k=candidate_k,
        rrf_k=config.rrf_k,
        reranker=None,                      # 重排要联网，离线模式不使用
    )


def run_pipeline(items: list[EvalItem], mode: str, top_k: int, candidate_k: int) -> tuple[list[dict], dict]:
    """对每条问题跑一次 RAG，收集 answer / contexts / 耗时

    返回 (records, run_env)：
      - records 会先存成 jsonl，之后重新打分可以直接 --reuse，不必再跑模型；
      - run_env 记录本次实际使用的「集合 + 嵌入后端 + 模型 + 维度」，
        写进报告 meta —— 不同维度/后端的指标不可比较，必须可追溯。
    """
    # 必须在建检索器之前覆盖：检索器是在 RagService.__init__ / build_offline_retriever 里建的
    config.retrieve_k = top_k
    config.candidate_k = candidate_k

    if mode == "mock":
        if config.rerank_enabled:
            print("[离线模式] 重排需要联网调用 DashScope，本轮自动关闭")
            config.rerank_enabled = False
        from offline import mock_answer
        retriever = build_offline_retriever(candidate_k)
        records = []
        for item in items:
            start = time.perf_counter()
            hits = retriever.invoke(item.question)
            contexts = [doc.page_content for doc in hits]
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
        return records, {
            "collection": "（离线替身：未使用向量库）",
            "embedding_backend": "（离线替身：无嵌入）",
            "embedding_model": None,
            "embedding_dim": None,
        }

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
    return records, {
        "collection": rag.collection_name,
        "embedding_backend": rag.profile.backend,
        "embedding_model": rag.profile.model,
        "embedding_dim": rag.embedding_dim,
    }


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

    judge_llm = build_judge_llm(judge_provider, judge_model)
    # 裁判侧的 embedding 跟随本次运行的嵌入后端：
    # local 时零成本、不消耗云端额度，且与检索侧保持同一向量空间，口径更一致
    judge_embeddings = LangchainEmbeddingsWrapper(embeddings_module.build_embeddings())

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
    )

    frame = result.to_pandas()
    per_question: list[dict] = []
    for record, (_, row) in zip(records, frame.iterrows()):
        per_question.append({**record, **{name: row.get(name) for name in METRIC_NAMES}})

    metrics = {name: (None if frame[name].isna().all() else float(frame[name].mean()))
               for name in METRIC_NAMES if name in frame.columns}
    return metrics, per_question


def judge_api_key_env(provider: str) -> str | None:
    """裁判来源对应的环境变量名（None = 不需要 key）"""
    if provider == "dashscope":
        return "DASHSCOPE_API_KEY"
    if provider in ("deepseek", "openai-compatible"):
        return config.openai_compatible_api_key_env
    return None                     # ollama 是本地服务；dashscope-native 走 SDK 自己的读取逻辑


def require_judge_key(provider: str) -> None:
    """在真正打分前确认裁判的 key 就位，并给出**准确的变量名**

    原来的实现写死了 DASHSCOPE_API_KEY，换成 DeepSeek 时会误报"缺 key"。
    """
    env_name = judge_api_key_env(provider)
    if env_name and not os.environ.get(env_name):
        raise SystemExit(
            f"裁判来源 {provider} 需要环境变量 {env_name}"
            f"（写进 .env 即可，该文件已在 .gitignore 中排除），或改用 --judge heuristic"
        )


def build_judge_llm(provider: str, model: str):
    """裁判模型。默认走 DashScope 的 OpenAI 兼容端点——这是 RAGAS 最稳的接法

    `deepseek` / `openai-compatible`：走 config 里那组 OpenAI 兼容配置，
    这样"生成用 A、裁判用 B"或"两者都用 DeepSeek"都能表达，且不写死厂商。

    ⚠️ 口径提醒：若**生成与裁判用同一个模型**，会引入"自我偏好"偏差。
    报告 meta 里同时记录 `chat_model` 与 `judge_model`，就是为了让这一点可见。
    """
    if provider == "dashscope":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
        )
    if provider in ("deepseek", "openai-compatible"):
        from langchain_openai import ChatOpenAI

        env_name = config.openai_compatible_api_key_env
        key = os.environ.get(env_name)
        if not key:
            raise SystemExit(f"裁判来源 {provider} 需要环境变量 {env_name}")
        # --judge-model 的默认值是给 dashscope 准备的（qwen-plus）；
        # 换到 OpenAI 兼容端点时若没显式指定，就用 config 里的模型名，避免拿 qwen-plus 去打 DeepSeek
        resolved = model if model and model != "qwen-plus" else config.openai_compatible_model
        return ChatOpenAI(
            model=resolved,
            api_key=key,
            base_url=config.openai_compatible_base_url,
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
    raise SystemExit(
        f"未知的裁判来源：{provider}"
        f"（可选 dashscope / deepseek / openai-compatible / ollama / dashscope-native）"
    )


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
def main(argv: list[str] | None = None) -> int:
    """argv 留出参数位，便于 experiments.py 以函数方式驱动多组实验"""
    parser = argparse.ArgumentParser(description="RAGAS 评测")
    parser.add_argument("--mode", choices=["real", "mock"], default="real",
                        help="real=真模型+RAGAS；mock=离线自检（不联网、不需要 Key）")
    parser.add_argument("--name", default="baseline", help="本轮实验名，用于文件命名（如 baseline / optimized）")
    parser.add_argument("--out", default=None, help="报告输出路径，默认 evaluation/report_<name>.json")
    parser.add_argument("--eval-set", default=str(EVAL_DIR / "eval_set.jsonl"), help="评测集路径")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 config.retrieve_k（阶段 3 做 A/B 用）")
    parser.add_argument("--retriever", choices=["vector", "bm25", "hybrid"], default=None,
                        help="检索策略：vector 纯向量 / bm25 纯关键词 / hybrid 混合（默认取 config）")
    parser.add_argument("--candidate-k", type=int, default=None,
                        help="粗排候选数（混合检索与重排从这批候选里精选）")
    parser.add_argument("--collection", default=None,
                        help="检索哪个向量库集合（分块对比实验用，配合 scripts/ingest.py --collection）")
    parser.add_argument("--embedding-backend", choices=["dashscope", "local"], default=None,
                        help="嵌入后端：dashscope 云端 / local 本地 bge-small-zh-v1.5（默认取 config）")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="分块参数：离线模式真实生效；real 模式只用于记录（检索不切分，参数由入库时决定）")
    parser.add_argument("--chunk-overlap", type=int, default=None,
                        help="分块重叠：同上。注意用 None 判断，0 是合法值")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                        help="开启 DashScope 重排（gte-rerank-v2）")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="关闭重排（默认）")
    parser.add_argument("--judge", choices=["ragas", "heuristic"], default=None,
                        help="裁判：ragas=真 RAGAS；heuristic=离线代理指标。real 模式默认 ragas")
    parser.add_argument("--judge-provider", default="dashscope",
                        choices=["dashscope", "deepseek", "openai-compatible", "ollama", "dashscope-native"],
                        help="裁判模型来源。deepseek/openai-compatible 走 config 里的 OpenAI 兼容配置")
    parser.add_argument("--judge-model", default="qwen-plus",
                        help="裁判模型名（打分用，别用太慢的模型）。用 deepseek 时留空/默认即可，会取 config 里的模型名")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，调试用")
    parser.add_argument("--reuse", default=None, help="复用已有的 runs jsonl，跳过 RAG 链路只重新打分")
    args = parser.parse_args(argv)

    judge = args.judge or ("heuristic" if args.mode == "mock" else "ragas")
    # 阶段 3：把命令行开关写回 config，检索器是在后面才构造的，所以这里覆盖有效
    if args.retriever:
        config.retrieval_method = args.retriever
    if args.rerank is not None:
        config.rerank_enabled = args.rerank
    if args.collection:
        config.collection_override = args.collection   # 显式指定（分块实验用），不再按后端派生命名
    if args.embedding_backend:
        config.embedding_backend = args.embedding_backend
    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:          # 0 是合法重叠度，所以用 is not None 判断
        config.chunk_overlap = args.chunk_overlap
    top_k = args.top_k if args.top_k is not None else config.retrieve_k
    candidate_k = args.candidate_k if args.candidate_k is not None else config.candidate_k
    if args.out:
        out_path = Path(args.out)
    elif args.mode == "mock":
        out_path = REPORTS_DIR / f"report_{args.name}_mock.json"
    else:
        out_path = EVAL_DIR / f"report_{args.name}.json"

    started = time.perf_counter()
    print("=" * 78)
    print(f"[评测] 实验名={args.name} 模式={args.mode} 裁判={judge}")
    print(f"[评测] 检索：策略={config.retrieval_method} "
          f"集合={embeddings_module.default_collection_name()} top_k={top_k} "
          f"候选={candidate_k} 重排={'开' if config.rerank_enabled else '关'}")
    print(f"[评测] 嵌入：后端={config.embedding_backend}（真实维度在准备阶段实测）")
    print(f"[评测] 分块：{config.chunk_size}/{config.chunk_overlap}")
    print("=" * 78)

    # 1) 拿答案与检索片段
    if args.reuse:
        reuse_path = Path(args.reuse)
        records = load_records(reuse_path)
        # 复用明细时并不知道当时的嵌入环境，如实标注为"未知"而不是冒充已知
        run_env = {"collection": "（复用明细：未知）", "embedding_backend": "（复用明细：未知）",
                   "embedding_model": None, "embedding_dim": None}
        print(f"[评测] 复用已有结果：{reuse_path}（{len(records)} 条），跳过 RAG 链路")
    else:
        items = load_eval_set(Path(args.eval_set))
        if args.limit:
            items = items[:args.limit]
        print(summarize(items))
        print("-" * 78)
        records, run_env = run_pipeline(items, args.mode, top_k, candidate_k)
        run_path = RUNS_DIR / f"{args.name}_{args.mode}.jsonl"
        save_records(run_path, records)
        print(f"[评测] 运行明细已保存：{run_path}")

    # 2) 打分
    print("-" * 78)
    if judge == "ragas":
        if args.mode == "mock" or args.reuse:
            # 这两种模式下 RAG 链路不跑云端，所以必须在这里显式确认裁判的 key 就位
            require_judge_key(args.judge_provider)
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
            # ★ 记下生成侧用的模型：若它与裁判模型相同，就是"自己给自己打分"（自我偏好偏差），
            #   必须让读报告的人看得见这一点，否则分数会被当成中立的
            "chat_backend": config.chat_backend,
            "chat_model": (config.chat_model_name if config.chat_backend == "dashscope"
                           else config.openai_compatible_model),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_seconds": round(elapsed, 2),
            "eval_set": str(Path(args.eval_set)),
            "eval_size": len(records),
            "retrieval": {
                "method": config.retrieval_method,
                "offline_stand_in": args.mode == "mock",
                "top_k": top_k,
                "candidate_k": candidate_k,
                "rrf_k": config.rrf_k if config.retrieval_method == "hybrid" else None,
                "rerank": config.rerank_enabled,
                "rerank_model": config.rerank_model if config.rerank_enabled else None,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                # 实际使用（run_env 来自真实的检索器对象，不是 config 推断）
                "collection": run_env["collection"],
                "embedding_backend": run_env["embedding_backend"],
                "embedding_model": run_env["embedding_model"],
                "embedding_dim": run_env["embedding_dim"],
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
