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

噪声地板（★ 下结论前必读）：
    向量腿跨进程**不可复现** —— chromadb 走 HNSW 近似检索，索引图在进程内重建，
    实测 36 题里有 24 题的**候选集合**跨进程就不同（不只是顺序）。
    因此同一配置在不同进程会给出不同的指标（实测 doc_hit@1 抖动 2 题 / 36）。
    → 用 `--repeat N` 起 N 个**独立进程**重复测量：报告取均值，并写入实测极差
      （`meta.noise_floor`），对比表会自动把落在噪声内的行标成"无法排序"。
    → 注意同进程内重复是**测不出**这个噪声的（实测同进程 0/36 差异），所以必须起独立进程。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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
from retrievers import BM25Retriever, DashScopeReranker, VectorRetriever, build_retriever, load_all_documents  # noqa: E402

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


def build_retriever_for(method: str, collection: str, top_k: int, candidate_k: int,
                        rerank: bool = False):
    """按检索策略构造检索器，并返回实际运行环境（写进报告 meta）"""
    profile, embeddings, dim = embeddings_module.prepare(config.embedding_backend)
    from vector_stores import VectorStoreService
    service = VectorStoreService(embedding=embeddings, collection_name=collection)
    embeddings_module.verify_and_stamp(service.vector_store._collection, profile, dim, collection)

    corpus = load_all_documents(service.vector_store)
    vector_leg = VectorRetriever(service, k=candidate_k) if method in ("vector", "hybrid") else None
    # 重排是可选项：无 Key / 网络不通时 DashScopeReranker 会**降级为粗排顺序**并打警告，
    # 不会让评测挂掉 —— 但那样测出来的就不是"带重排"的效果，跑完必须看日志确认没降级。
    reranker = DashScopeReranker() if rerank else None
    retriever = build_retriever(
        method, vector_leg=vector_leg, documents=corpus, k=top_k,
        candidate_k=candidate_k, rrf_k=config.rrf_k, reranker=reranker,
    )
    run_env = {
        "collection": collection,
        "embedding_backend": profile.backend,
        "embedding_model": profile.model,
        "embedding_dim": dim,
        "corpus_chunks": len(corpus),
        "rerank": bool(rerank),
    }
    return retriever, run_env


def load_chunks(collection: str):
    """一次性读出集合里全部片段的 id / 文本 / 向量（两种片段级口径共用）"""
    client = chromadb.Client(CHROMA_SETTINGS)
    raw = client.get_collection(collection).get(include=["documents", "metadatas", "embeddings"])
    chunk_ids: list[str] = []
    for meta in raw["metadatas"]:
        source = (meta or {}).get("source", "?")
        index = (meta or {}).get("chunk_index", "?")
        chunk_ids.append(f"{source}#{index}")
    return chunk_ids, list(raw["documents"]), [list(v) for v in raw["embeddings"]]


def build_weak_labels(items: list[EvalItem], embeddings, threshold: float,
                      chunk_ids: list[str], chunk_vectors: list[list[float]]) -> tuple[list[set[str]], dict]:
    """片段级**弱标注**：正例 = 与标准答案嵌入相似度 ≥ 阈值的片段

    向量的取法：直接读 Chroma 里已存的向量（不再重复调用嵌入），只对 ground_truth 做嵌入。
    """
    truth_vectors = embeddings.embed_documents([item.ground_truth for item in items])

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


def build_keyword_labels(items: list[EvalItem], chunk_ids: list[str],
                         chunk_texts: list[str]) -> tuple[list[set[str]], dict]:
    """片段级**对抗性口径**：正例 = 片段文本里出现了该问题的 answer_keywords（纯词面匹配）

    ★ 这个口径是刻意设计成"对向量检索不利"的：
      关键词是字面匹配，所以 BM25 这类词面检索天然占优；向量检索召回的往往是
      "语义相近但用词不同"的片段，在这个口径下会被判成不相关。
      它的价值在于**换个方向施压**：
        - 若向量检索在此口径下仍然胜出 → 结论比弱标注口径下硬得多；
        - 若名次反转 → 说明弱标注口径下的优势确实来自"标注生成器与向量检索同源"的循环性。
      两种结果都有信息量，这也是为什么两个口径要一起给。
    """
    relevants: list[set[str]] = []
    density: list[int] = []
    no_positive: list[str] = []
    for item in items:
        keywords = [kw for kw in item.answer_keywords if kw]
        positive = set()
        if keywords:
            for chunk_id, text in zip(chunk_ids, chunk_texts):
                if any(kw in text for kw in keywords):
                    positive.add(chunk_id)
        if not positive:
            # 关键词一个都没命中的情况必须显式记下来：该 query 的指标会恒为 0，
            # 如果这种样本变多，说明评测集关键词与语料脱节了（是数据问题，不是检索问题）
            no_positive.append(item.id)
        relevants.append(positive)
        density.append(len(positive))

    stats = {
        "chunk_total": len(chunk_ids),
        "avg_positive_per_query": round(sum(density) / len(density), 2) if density else 0.0,
        "items_without_positive": no_positive,
        "note": ("片段级对抗性口径：正例 = 片段包含该问题的 answer_keywords（纯词面），"
                 "刻意偏袒关键词检索；若向量检索在此口径下仍胜出，结论更硬。"),
    }
    return relevants, stats


# --------------------------------------------------------------------------- 单组评测
def evaluate_one(items: list[EvalItem], method: str, collection: str, top_k: int,
                 candidate_k: int, threshold: float,
                 rerank: bool = False) -> tuple[dict, list[dict], dict, dict]:
    retriever, run_env = build_retriever_for(method, collection, top_k, candidate_k, rerank)

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

    chunk_ids, chunk_texts, chunk_vectors = load_chunks(collection)
    corpus_size = len(chunk_ids)
    caliber: dict = {
        "doc": {"label_level": "文档级", "chunk_total": corpus_size,
                "text": ir_metrics.caliber_text("doc")},
    }

    # 片段级口径一：弱标注（嵌入相似度）—— 与向量检索同源，存在循环性；lift 只解决跨分组可比
    if threshold > 0:
        _profile, embeddings, _dim = embeddings_module.prepare(config.embedding_backend)
        chunk_relevants, chunk_stats = build_weak_labels(
            items, embeddings, threshold, chunk_ids, chunk_vectors
        )
        metrics.update(ir_metrics.evaluate_rankings(chunk_rankings, chunk_relevants, K_VALUES, prefix="chunk"))
        base = ir_metrics.base_rate(chunk_relevants, corpus_size)
        ir_metrics.add_lift(metrics, "chunk", base)
        caliber["chunk"] = {"label_level": "片段级（弱标注）", "base_rate": round(base, 4),
                            "text": ir_metrics.caliber_text("chunk"), **chunk_stats}
        for record, relevant in zip(per_question, chunk_relevants):
            record["label_level"] = "chunk(weak)"
            record["chunk_relevant_n"] = len(relevant)

    # 片段级口径二：对抗性口径（答案关键词词面命中）—— 刻意偏袒关键词检索，用来反向施压
    kw_relevants, kw_stats = build_keyword_labels(items, chunk_ids, chunk_texts)
    if kw_stats["avg_positive_per_query"] > 0:
        metrics.update(ir_metrics.evaluate_rankings(chunk_rankings, kw_relevants, K_VALUES, prefix="kw"))
        kw_base = ir_metrics.base_rate(kw_relevants, corpus_size)
        ir_metrics.add_lift(metrics, "kw", kw_base)
        caliber["kw"] = {"label_level": "片段级（对抗性口径）", "base_rate": round(kw_base, 4),
                         "text": ir_metrics.caliber_text("kw"), **kw_stats}
        for record, relevant in zip(per_question, kw_relevants):
            record["kw_relevant_n"] = len(relevant)

    run_env["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    return metrics, per_question, run_env, caliber


# headline 噪声地板只看这些"下结论时真正会引用的主指标"（详见 merge_reports 里的说明）
PRIMARY_NOISE_METRICS = ("doc_hit_rate@1", "doc_mrr", "kw_precision@3", "chunk_precision@3")


def merge_reports(paths: list[Path], merged_name: str) -> Path:
    """把 N 份重复报告合并成一份：指标取均值，并写入实测极差（该配置的噪声地板）

    ★ 噪声地板是**每个配置各自的属性**，不是全局常数：
      候选 10 时近并列的候选集较小、候选 32 时更大，地板可能不同。
      所以这里对每个配置分别测量、分别记录，不做任何跨配置复用。
    """
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    n = len(runs)
    means: dict[str, float | None] = {}
    spreads: dict[str, float] = {}
    for name in runs[0]["metrics"]:
        vals = [float(r["metrics"][name]) for r in runs if r["metrics"].get(name) is not None]
        means[name] = round(sum(vals) / len(vals), 4) if vals else None
        spreads[name] = round(max(vals) - min(vals), 4) if len(vals) > 1 else 0.0

    eval_size = runs[0]["meta"]["eval_size"]
    # ★ 不能对所有指标取极差的最大值当统一地板：lift 的量纲是 4~8、hit_rate 是 0~1，
    #   混在一起算会得到"11.5 题"这种荒唐数（冒烟实测踩到过）。
    #   对外的 headline 只取"下结论时真正引用的主指标"，逐指标极差放 by_metric 供对比表按行取用。
    present = [spreads[m] for m in PRIMARY_NOISE_METRICS if m in spreads]
    headline = max(present) if present else (max(spreads.values()) if spreads else 0.0)

    merged = dict(runs[0])
    merged["meta"] = dict(runs[0]["meta"])
    merged["meta"]["name"] = merged_name
    merged["meta"]["repeat_n"] = n
    merged["meta"]["repeat_runs"] = [r["meta"]["name"] for r in runs]
    merged["meta"]["noise_floor"] = {
        "value": headline,                                  # 主指标口径（0~1 尺度）
        "items": round(headline * eval_size, 1),            # 换算成"题数"，可操作单位
        "eval_size": eval_size,
        "headline_metrics": [m for m in PRIMARY_NOISE_METRICS if m in spreads],
        "max_by_metric": max(spreads.values()) if spreads else 0.0,
        "by_metric": spreads,                               # 逐指标极差：对比表按行取用
        "note": ("同一配置 N 个独立进程实测极差。向量腿为 HNSW 近似检索，"
                 "候选集合跨进程会变（实测 36 题里 24 题的集合不同），故存在此下界。"
                 "对比表按 by_metric 逐行判断：|差值| 不超过该指标地板的一律标注为无法排序。"
                 f"headline 只取主指标 {list(PRIMARY_NOISE_METRICS)} 里的最大值。"),
    }
    merged["meta"]["per_question_note"] = (
        f"per_question 取自第 1 次运行（共 {n} 次，各次排名可能略有差异）"
    )
    merged["meta"]["warning"] = (
        f"IR 指标（确定性，非 RAGAS）：测检索排序质量，不含生成质量。"
        f"doc_* 文档级（无循环）/ kw_* 对抗口径（无嵌入）/ chunk_* 弱标注（嵌入同源）。"
        f"本报告为 {n} 个独立进程的均值，实测噪声地板 {headline:.4f}"
        f"（{round(headline * eval_size, 1)} 题 / {eval_size}）。"
    )
    merged["metrics"] = means

    out = REPORTS_DIR / f"report_{merged_name}_ir.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[IR] 合并 {n} 次 -> {out.name}"
          f"（主指标噪声地板 {headline:.4f} = {round(headline * eval_size, 1)} 题 / {eval_size}；"
          f"任一指标最大极差 {max(spreads.values()) if spreads else 0.0:.4f}）")
    return out


def run_repeats(args, configs, methods, backend: str, top_k: int, candidate_k: int) -> list[Path]:
    """起 N 个**独立进程**重复跑同一批配置，返回合并后的报告路径

    为什么必须是独立进程（subprocess 而不是 multiprocessing / 循环调用）：
        噪声来自 chromadb 的 HNSW 近似检索，索引图在**每个进程里重建**，
        于是"近似并列"的候选**集合**都会变。同一个进程内重复跑是测不出来的
        —— 实测同进程两次 0/36 题差异，跨进程 24/36 题集合不同。
        subprocess 也与我们手工测量噪声时的方式完全一致（可复现）。
    """
    n = args.repeat
    print(f"[IR] 重复测量：{n} 个独立进程（指标取均值，同时记录实测噪声地板）")
    for i in range(1, n + 1):
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--name", f"{args.name}_r{i}",
               "--configs", args.configs,
               "--top-k", str(top_k),
               "--candidate-k", str(candidate_k),
               "--embedding-backend", backend,
               "--sim-threshold", str(args.sim_threshold),
               "--eval-set", args.eval_set,
               "--repeat", "1"]
        if args.retrievers:
            cmd += ["--retrievers", args.retrievers]
        if args.retriever:
            cmd += ["--retriever", args.retriever]
        if args.corpus:
            cmd += ["--corpus", args.corpus]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.rerank:
            cmd.append("--rerank")             # 重排开关必须显式传给子进程，否则重复测量里它悄悄关掉
        if args.no_ingest or i > 1:
            cmd.append("--no-ingest")           # 第 1 个进程负责入库，其余复用同一集合
        print(f"\n[IR] --- 第 {i}/{n} 次（独立进程）---")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"[IR] 第 {i} 次运行失败（退出码 {proc.returncode}），中止重复测量")
            return []

    merged: list[Path] = []
    for chunk_size, chunk_overlap in configs:
        for method in methods:
            suffix = f"_{method}" if method else ""
            tail = f"{backend}_c{chunk_size}_o{chunk_overlap}{suffix}"
            paths = [REPORTS_DIR / f"report_{args.name}_r{i}_{tail}_ir.json" for i in range(1, n + 1)]
            missing = [p.name for p in paths if not p.exists()]
            if missing:
                print(f"[IR] 缺少重复报告，跳过该组合并：{missing}")
                continue
            merged.append(merge_reports(paths, f"{args.name}_{tail}"))
    return merged


def finalize(reports: list[Path], name: str, backend: str, started: float) -> int:
    """渲染对比表并打印（单次与重复测量两条路径共用）"""
    if not reports:
        print("[IR] 没有任何成功的实验组")
        return 1

    loaded = [json.loads(Path(p).read_text(encoding="utf-8")) for p in reports]
    labels = [r["meta"]["name"] for r in loaded]
    text = render(loaded, labels, worst=0)
    compare_path = REPORTS_DIR / f"{name}_{backend}_ir_compare.md"
    compare_path.write_text(text, encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(text)
    print(f"[IR] 共 {len(reports)} 组，总耗时 {time.perf_counter() - started:.1f}s")
    print(f"[IR] 对比表已写入：{compare_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="IR 指标评测（确定性，不需要 LLM 裁判）")
    parser.add_argument("--name", default="ir", help="实验名前缀")
    parser.add_argument("--configs", default="1000/100", help="分块参数列表，如 150/15,300/30")
    parser.add_argument("--retrievers", default=None, help="检索策略列表，如 vector,bm25,hybrid")
    parser.add_argument("--retriever", default=None, help="只跑单一检索策略")
    parser.add_argument("--top-k", type=int, default=None, help="交给下游的片段数（对比时必须固定）")
    parser.add_argument("--candidate-k", type=int, default=None, help="粗排候选数")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                        help="开启 DashScope 重排（gte-rerank-v2）；需 DASHSCOPE_API_KEY")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="关闭重排（默认）")
    parser.add_argument("--embedding-backend", choices=["dashscope", "local"], default=None)
    # ★ 不要收紧这个阈值（默认 0.6 是刻意保留的，不是"没调好"）
    #   收紧的收益：基准率下降 → lift 的区分度上升；
    #   收紧的代价：留下来的正例恰好是嵌入模型最"自信"的那批 → 与标注生成器同源更强 → 循环性加重。
    #   而"比检索器"时三路的基准率本来就完全相同（同一配置、同一套标注），
    #   收紧只能加重循环性、换不到任何可比性收益 —— 只付代价、无收益。
    #   跨分块配置的可比性问题由 lift = precision@k / 基准率 解决（见 ir_metrics.add_lift）。
    parser.add_argument("--sim-threshold", type=float, default=0.6,
                        help="片段级弱标注的相似度阈值；填 0 关闭该口径。★ 不建议收紧，理由见代码注释")
    parser.add_argument("--eval-set", default=str(EVAL_DIR / "eval_set.jsonl"))
    parser.add_argument("--corpus", default=None,
                        help="语料目录（默认 assets；扩语料后用 corpus，集合名会自动带 _corpus 标记）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，调试用")
    parser.add_argument("--no-ingest", action="store_true", help="跳过入库（复用已有集合）")
    # ★ 默认 3 次：单次运行的指标会被噪声地板（实测 2 题 / 36）淹没，见模块 docstring
    parser.add_argument("--repeat", type=int, default=3,
                        help="起 N 个独立进程重复测量，报告取均值并记录该配置的噪声地板（默认 3；1 = 不重复）")
    args = parser.parse_args()

    from experiments import parse_configs, parse_methods, corpus_tag_of
    configs = parse_configs(args.configs)
    methods = parse_methods(args.retrievers, args.retriever)
    corpus_dir = Path(args.corpus).resolve() if args.corpus else None
    corpus_tag = corpus_tag_of(corpus_dir)
    backend = args.embedding_backend or config.embedding_backend
    config.embedding_backend = backend
    top_k = args.top_k if args.top_k is not None else config.retrieve_k
    candidate_k = args.candidate_k if args.candidate_k is not None else config.candidate_k
    rerank = config.rerank_enabled if args.rerank is None else args.rerank
    if rerank and not os.environ.get("DASHSCOPE_API_KEY"):
        print("[IR] 警告：开启了重排但没有 DASHSCOPE_API_KEY —— DashScopeReranker 会降级为粗排顺序，"
              "本轮数字并不代表「带重排」的效果。")

    items = load_eval_set(Path(args.eval_set))
    if args.limit:
        items = items[:args.limit]

    print("=" * 78)
    print(f"[IR] 实验名={args.name} 嵌入后端={backend} top_k={top_k} 候选={candidate_k}")
    print(f"[IR] 分块维度：{'，'.join(f'{s}/{o}' for s, o in configs)}")
    print(f"[IR] 检索维度：{'，'.join(m or '（默认）' for m in methods)}")
    print(f"[IR] 语料目录：{corpus_dir or 'assets（默认）'}  集合标记：{corpus_tag or '（无）'}")
    print(f"[IR] 片段级弱标注阈值：{args.sim_threshold}（0 = 关闭）")
    print(f"[IR] 重排：{'开（' + config.rerank_model + '）' if rerank else '关'}")
    print(summarize(items), sep="\n")
    print("=" * 78)

    reports: list[Path] = []
    started = time.perf_counter()

    # 重复测量走独立进程（父进程只负责编排与合并），与手工测量噪声的方式一致
    if args.repeat > 1:
        return finalize(
            run_repeats(args, configs, methods, backend, top_k, candidate_k),
            args.name, backend, started,
        )

    ingested: set[tuple] = set()

    for chunk_size, chunk_overlap in configs:
        collection = collection_name_for(chunk_size, chunk_overlap, backend, corpus_tag)
        if not args.no_ingest and (chunk_size, chunk_overlap, backend) not in ingested:
            print(f"\n[IR] 入库到集合 {collection}（分块 {chunk_size}/{chunk_overlap}）")
            ingest_collection(chunk_size, chunk_overlap, backend, corpus_dir)
            ingested.add((chunk_size, chunk_overlap, backend))

        for method in methods:
            run_name = (f"{args.name}_{backend}_c{chunk_size}_o{chunk_overlap}"
                        + (f"_{method}" if method else ""))
            print(f"\n{'-' * 78}\n[IR] 评测 {run_name}\n{'-' * 78}")
            metrics, per_question, run_env, caliber = evaluate_one(
                items, method, collection, top_k, candidate_k, args.sim_threshold, rerank
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
                        "rerank": run_env["rerank"],
                        "rerank_model": config.rerank_model if run_env["rerank"] else None,
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
                    "repeat_n": 1,          # 单次运行：无法给出该配置的噪声地板，对比时不做"无法排序"标注
                    "warning": ("IR 指标（确定性，非 RAGAS）：测的是检索排序质量，不含生成质量。"
                                "doc_* 文档级（无循环）/ kw_* 对抗口径（无嵌入）/ chunk_* 弱标注（嵌入同源）。"
                                "单次运行未测噪声地板，结论前请用 --repeat 3。"),
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
