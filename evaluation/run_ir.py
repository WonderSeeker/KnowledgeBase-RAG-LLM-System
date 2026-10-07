"""用 IR 指标评测**检索本身**（确定性，不需要 LLM 裁判）

为什么这是本项目"效果结论"的出路：
    RAGAS 的 4 个指标要 LLM 裁判，没有可用 API Key 就跑不起来。
    而「检索策略 / 分块参数」的效果**只取决于检索结果本身** —— 用确定性指标就能测。
    所以这是把 CHANGES 4.3 / 4.4 那两张"替身数据表"变成真实结论的唯一手段。

两种标注口径（同一份报告里都输出，用前缀区分，报告里逐条写明）：

    doc_*    文档级 —— 正例 = 评测集的 `relevant_docs`（应命中的 .txt）。
             规则确定、完全可复现，是**主口径**。
             注意：每个 query 只有 1 个正例时，`doc_recall@k` 与 `doc_hit_rate@k` 数值必然相同。

    chunk_*  片段级 —— **弱标注**：正例 = 与标准答案嵌入相似度 ≥ 阈值的片段。
             ★ 这是自动生成的弱标注，不是人工标注；阈值本身也是口径的一部分，会写进报告。

用法：
    # 4.3 检索策略对比（★ 必须固定 k，否则不可比）
    .\\.venv\\Scripts\\python.exe evaluation\\run_ir.py --name ir_retriever ^
        --configs 1000/100 --retrievers vector,bm25,hybrid --top-k 4 --candidate-k 10

    # 4.4 分块参数对比（固定检索策略与 k）
    .\\.venv\\Scripts\\python.exe evaluation\\run_ir.py --name ir_chunk ^
        --configs 150/15,300/30,600/60,1000/100 --retriever hybrid --top-k 4 --candidate-k 32
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _path in (str(EVAL_DIR), str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import chromadb                                                    # noqa: E402
from langchain_core.documents import Document                      # noqa: E402

import config_data as config                                       # noqa: E402
import embeddings as embeddings_module                             # noqa: E402
import ir_metrics                                                  # noqa: E402
from chroma_quiet import CHROMA_SETTINGS                           # noqa: E402
from dataset import EvalItem, load_eval_set, summarize             # noqa: E402
from experiments import collection_name_for, ingest_collection      # noqa: E402
from report import render                                          # noqa: E402
from retrievers import BM25Retriever, VectorRetriever, build_retriever, load_all_documents  # noqa: E402

REPORTS_DIR = EVAL_DIR / "reports"
K_VALUES = (1, 3, 5)


# --------------------------------------------------------------------------- 工具
def chunk_id_of(doc: Document) -> str:
    """片段级 id：来源文件 + 段内序号（chunk_index 由入库时写入 metadata）"""
    source = doc.metadata.get("source") or "?"
    index = doc.metadata.get("chunk_index")
    return f"{source}#{index if index is not None else '?'}"


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（本地 bge 输出已 L2 归一化，但仍显式归一化，避免对后端做假设）"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def build_retriever_for(method: str, collection: str, top_k: int, candidate_k: int):
    """按检索策略构造检索器，并返回实际运行环境（写进报告 meta）"""
    profile, embeddings, dim = embeddings_module.prepare(config.embedding_backend)
    from vector_stores import VectorStoreService
    service = VectorStoreService(embedding=embeddings, collection_name=collection)
    embeddings_module.verify_and_stamp(service.vector_store._collection, profile, dim, collection)

    corpus = load_all_documents(service.vector_store)
    vector_leg = VectorRetriever(service, k=candidate_k) if method in ("vector", "hybrid") else None
    retriever = build_retriever(
        method, vector_leg=vector_leg, documents=corpus, k=top_k,
        candidate_k=candidate_k, rrf_k=config.rrf_k, reranker=None,
    )
    run_env = {
        "collection": collection,
        "embedding_backend": profile.backend,
        "embedding_model": profile.model,
        "embedding_dim": dim,
        "corpus_chunks": len(corpus),
    }
    return retriever, run_env


def build_weak_labels(collection: str, profile, embeddings, items: list[EvalItem],
                      threshold: float) -> tuple[list[set[str]], dict]:
    """片段级**弱标注**：正例 = 与标准答案嵌入相似度 ≥ 阈值的片段

    向量的取法：直接读 Chroma 里已存的向量（不再重复调用嵌入），只对 ground_truth 做嵌入。
    """
    client = chromadb.Client(CHROMA_SETTINGS)
    raw = client.get_collection(collection).get(include=["documents", "metadatas", "embeddings"])

    chunk_ids: list[str] = []
    chunk_vectors: list[list[float]] = []
    for text, meta, vector in zip(raw["documents"], raw["metadatas"], raw["embeddings"]):
        source = (meta or {}).get("source", "?")
        index = (meta or {}).get("chunk_index", "?")
        chunk_ids.append(f"{source}#{index}")
        chunk_vectors.append(list(vector))

    truths = [item.ground_truth for item in items]
    truth_vectors = embeddings.embed_documents(truths)

    relevants: list[set[str]] = []
    density: list[int] = []
    for truth_vector in truth_vectors:
        scores = [cosine(truth_vector, chunk_vector) for chunk_vector in chunk_vectors]
        positive = {chunk_ids[i] for i, score in enumerate(scores) if score >= threshold}
        # 兜底：阈值过高导致某个 query 一个正例都没有时，取最相似的那个，
        # 否则该 query 的 recall/ndcg 恒为 0，会整体拉低均值而且看不出原因
        if not positive and scores:
            positive = {chunk_ids[max(range(len(scores)), key=lambda i: scores[i])]}
        relevants.append(positive)
        density.append(len(positive))

    stats = {
        "threshold": threshold,
        "chunk_total": len(chunk_ids),
        "avg_positive_per_query": round(sum(density) / len(density), 2) if density else 0.0,
        "note": ("片段级弱标注：正例 = 与标准答案嵌入相似度 ≥ 阈值的片段，"
                 "自动生成、非人工标注；阈值是口径的一部分。"),
    }
    return relevants, stats


# --------------------------------------------------------------------------- 单组评测
def evaluate_one(items: list[EvalItem], method: str, collection: str, top_k: int,
                 candidate_k: int, threshold: float) -> tuple[dict, list[dict], dict, dict]:
    retriever, run_env = build_retriever_for(method, collection, top_k, candidate_k)

    doc_rankings: list[list[str]] = []
    chunk_rankings: list[list[str]] = []
    per_question: list[dict] = []
    started = time.perf_counter()

    for item in items:
        query_start = time.perf_counter()
        hits = retriever.invoke(item.question)
        contexts = [doc.page_content for doc in hits]

        ranked_docs = ir_metrics.dedupe_keep_order(
            [doc.metadata.get("source") or "" for doc in hits]
        )
        ranked_chunks = [chunk_id_of(doc) for doc in hits]
        relevant_docs = set(item.relevant_docs)

        doc_rankings.append(ranked_docs)
        chunk_rankings.append(ranked_chunks)
        per_question.append({
            "id": item.id,
            "question": item.question,
            "relevant_docs": sorted(relevant_docs),
            "ranked_docs": ranked_docs,
            "ranked_chunks": ranked_chunks,
            "n_contexts": len(contexts),
            "doc_hit@1": ir_metrics.hit_rate_at_k(ranked_docs, relevant_docs, 1),
            "doc_rr": round(ir_metrics.reciprocal_rank(ranked_docs, relevant_docs), 4),
            "seconds": round(time.perf_counter() - query_start, 3),
        })

    doc_relevants = [set(item.relevant_docs) for item in items]
    metrics = ir_metrics.evaluate_rankings(doc_rankings, doc_relevants, K_VALUES, prefix="doc")

    caliber = {"doc": {"label_level": "文档级", "text": ir_metrics.caliber_text("doc")}}
    chunk_stats = None
    if threshold > 0:
        profile, embeddings, _ = embeddings_module.prepare(config.embedding_backend)
        chunk_relevants, chunk_stats = build_weak_labels(collection, profile, embeddings, items, threshold)
        metrics.update(ir_metrics.evaluate_rankings(chunk_rankings, chunk_relevants, K_VALUES, prefix="chunk"))
        caliber["chunk"] = {"label_level": "片段级（弱标注）", "text": ir_metrics.caliber_text("chunk"),
                            **chunk_stats}
        for record, relevant in zip(per_question, chunk_relevants):
            record["label_level"] = "chunk(weak)"
            record["chunk_relevant_n"] = len(relevant)

    run_env["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    return metrics, per_question, run_env, caliber


def main() -> int:
    parser = argparse.ArgumentParser(description="IR 指标评测（确定性，不需要 LLM 裁判）")
    parser.add_argument("--name", default="ir", help="实验名前缀")
    parser.add_argument("--configs", default="1000/100", help="分块参数列表，如 150/15,300/30")
    parser.add_argument("--retrievers", default=None, help="检索策略列表，如 vector,bm25,hybrid")
    parser.add_argument("--retriever", default=None, help="只跑单一检索策略")
    parser.add_argument("--top-k", type=int, default=None, help="交给下游的片段数（对比时必须固定）")
    parser.add_argument("--candidate-k", type=int, default=None, help="粗排候选数")
    parser.add_argument("--embedding-backend", choices=["dashscope", "local"], default=None)
    parser.add_argument("--sim-threshold", type=float, default=0.6,
                        help="片段级弱标注的相似度阈值；填 0 关闭片段级口径")
    parser.add_argument("--eval-set", default=str(EVAL_DIR / "eval_set.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，调试用")
    parser.add_argument("--no-ingest", action="store_true", help="跳过入库（复用已有集合）")
    args = parser.parse_args()

    from experiments import parse_configs, parse_methods
    configs = parse_configs(args.configs)
    methods = parse_methods(args.retrievers, args.retriever)
    backend = args.embedding_backend or config.embedding_backend
    config.embedding_backend = backend
    top_k = args.top_k if args.top_k is not None else config.retrieve_k
    candidate_k = args.candidate_k if args.candidate_k is not None else config.candidate_k

    items = load_eval_set(Path(args.eval_set))
    if args.limit:
        items = items[:args.limit]

    print("=" * 78)
    print(f"[IR] 实验名={args.name} 嵌入后端={backend} top_k={top_k} 候选={candidate_k}")
    print(f"[IR] 分块维度：{'，'.join(f'{s}/{o}' for s, o in configs)}")
    print(f"[IR] 检索维度：{'，'.join(m or '（默认）' for m in methods)}")
    print(f"[IR] 片段级弱标注阈值：{args.sim_threshold}（0 = 关闭）")
    print(summarize(items), sep="\n")
    print("=" * 78)

    reports: list[Path] = []
    started = time.perf_counter()
    ingested: set[tuple] = set()

    for chunk_size, chunk_overlap in configs:
        collection = collection_name_for(chunk_size, chunk_overlap, backend)
        if not args.no_ingest and (chunk_size, chunk_overlap, backend) not in ingested:
            print(f"\n[IR] 入库到集合 {collection}（分块 {chunk_size}/{chunk_overlap}）")
            ingest_collection(chunk_size, chunk_overlap, backend)
            ingested.add((chunk_size, chunk_overlap, backend))

        for method in methods:
            run_name = (f"{args.name}_{backend}_c{chunk_size}_o{chunk_overlap}"
                        + (f"_{method}" if method else ""))
            print(f"\n{'-' * 78}\n[IR] 评测 {run_name}\n{'-' * 78}")
            metrics, per_question, run_env, caliber = evaluate_one(
                items, method, collection, top_k, candidate_k, args.sim_threshold
            )

            report = {
                "meta": {
                    "name": run_name,
                    "mode": "ir",
                    "judge": "deterministic",
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "eval_set": str(Path(args.eval_set)),
                    "eval_size": len(items),
                    "retrieval": {
                        "method": method,
                        "top_k": top_k,
                        "candidate_k": candidate_k,
                        "rrf_k": config.rrf_k if method == "hybrid" else None,
                        "rerank": False,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "corpus_chunks": run_env["corpus_chunks"],
                        "collection": run_env["collection"],
                        "embedding_backend": run_env["embedding_backend"],
                        "embedding_model": run_env["embedding_model"],
                        "embedding_dim": run_env["embedding_dim"],
                    },
                    "caliber": caliber,
                    "avg_n_contexts": round(sum(r["n_contexts"] for r in per_question) / len(per_question), 2),
                    "avg_query_seconds": round(
                        sum(r["seconds"] for r in per_question) / len(per_question), 4),
                    "run_seconds": run_env["elapsed_seconds"],
                    "warning": ("IR 指标（确定性，非 RAGAS）：测的是检索排序质量，不含生成质量。"
                                "chunk_* 前缀为片段级弱标注口径。"),
                },
                "metrics": metrics,
                "per_question": per_question,
            }

            out_path = REPORTS_DIR / f"report_{run_name}_ir.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            reports.append(out_path)

            for name, value in metrics.items():
                print(f"    {name:<18} {value}")
            print(f"    -> {out_path.name}")

    if not reports:
        print("[IR] 没有任何成功的实验组")
        return 1

    import json
    loaded = [json.loads(Path(p).read_text(encoding="utf-8")) for p in reports]
    labels = [r["meta"]["name"] for r in loaded]
    text = render(loaded, labels, worst=0)
    compare_path = REPORTS_DIR / f"{args.name}_{backend}_ir_compare.md"
    compare_path.write_text(text, encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(text)
    print(f"[IR] 共 {len(reports)} 组，总耗时 {time.perf_counter() - started:.1f}s")
    print(f"[IR] 对比表已写入：{compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
