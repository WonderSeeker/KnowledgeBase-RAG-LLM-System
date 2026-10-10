"""信息检索指标：Hit Rate@k / Recall@k / MRR / nDCG@k（确定性，**不需要 LLM 裁判**）

为什么必须有它：
    RAGAS 的 4 个指标依赖 LLM 裁判；在没有可用 API Key 时跑不起来。
    而「检索策略 / 分块参数」的效果**只取决于检索结果本身**，用确定性指标就能测 ——
    这是把那两张"替身数据表"变成真实结论的唯一手段。

★ 标注口径（必须与数字一起输出，否则数字没有意义）

    【文档级】默认、主口径
        每个 query 的正例 = 评测集里的 `source` 字段（应命中的那个 .txt）。
        规则简单、完全可复现、零主观判断。
        局限：只有 3 份语料，且"答案跨多文档"的 query 会被低估。

    【片段级】可选、**弱标注**
        正例 = 与标准答案的嵌入相似度 > 阈值的片段。
        ★ 这是自动生成的弱标注，**不是人工标注**，报告里必须原样写明。

    报告里的 `caliber` 字段会记录实际用的是哪种口径 + n（样本数）。
"""

from __future__ import annotations

import math

DEFAULT_K_VALUES = (1, 3, 5)


def hit_rate_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """top-k 里是否至少命中一个正例（0/1）"""
    if not relevant:
        return 0.0
    return 1.0 if set(ranking[:k]) & relevant else 0.0


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """top-k 覆盖了正例集合的多大比例"""
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """top-k 里有多大比例是正例（噪声视角）

    ★ 为什么它比 recall 更适合跨"分块配置"比较：
      recall@k 会被语料粒度污染 —— 语料只有 6 个片段时，top-4 已经覆盖了 67% 的语料，
      recall 自然虚高；语料切成 32 个片段时 recall 天然偏低。
      这不是"检索变好/变差"，而是指标与语料规模耦合。
      precision@k 只问"top-k 里有多少是相关的"，与语料总片段数无关，因此可横向比。
    """
    if k <= 0:
        return 0.0
    return len(set(ranking[:k]) & relevant) / k


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    """第一个正例的排名倒数（1/rank）；没命中记 0。对"排得够不够前"最敏感"""
    for index, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """二元相关性的 nDCG@k

    DCG  = Σ rel_i / log2(i+1)
    IDCG = 正例全部排在最前时的 DCG
    """
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(index + 1)
              for index, doc_id in enumerate(ranking[:k], start=1) if doc_id in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_rankings(rankings: list[list[str]], relevants: list[set[str]],
                      k_values: tuple[int, ...] = DEFAULT_K_VALUES,
                      prefix: str = "") -> dict:
    """对一批 query 求平均指标

    :param rankings: 每个 query 的**有序**结果 id 列表（top1 在最前）
    :param relevants: 每个 query 的正例 id 集合
    :param prefix: 指标名前缀（如 "doc" / "chunk"），用于在同一份报告里区分两种口径
    :return: 扁平化的指标字典
    """
    if len(rankings) != len(relevants):
        raise ValueError("rankings 与 relevants 长度必须一致")
    n = len(rankings)
    if n == 0:
        return {}

    tag = f"{prefix}_" if prefix else ""
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"{tag}hit_rate@{k}"] = sum(
            hit_rate_at_k(r, rel, k) for r, rel in zip(rankings, relevants)) / n
        metrics[f"{tag}recall@{k}"] = sum(
            recall_at_k(r, rel, k) for r, rel in zip(rankings, relevants)) / n
        metrics[f"{tag}precision@{k}"] = sum(
            precision_at_k(r, rel, k) for r, rel in zip(rankings, relevants)) / n
    metrics[f"{tag}mrr"] = sum(reciprocal_rank(r, rel) for r, rel in zip(rankings, relevants)) / n
    metrics[f"{tag}ndcg@{max(k_values)}"] = sum(
        ndcg_at_k(r, rel, max(k_values)) for r, rel in zip(rankings, relevants)
    ) / n
    return {name: round(value, 4) for name, value in metrics.items()}


def base_rate(relevants: list[set[str]], corpus_size: int) -> float:
    """基准率：随机从语料里取一个片段，它恰好是正例的概率

    = 平均正例数 / 语料片段总数。

    ★ 为什么必须把它算出来并公开展示：
      `precision@k` 的**下界就是这个基准率**。不同分块配置下，基准率会随粒度漂移
      （实测：150/15 是 9.0%、1000/100 是 15.1%，相差 1.7 倍），
      于是"precision 变高"里有多少来自检索变好、有多少只是基准率涨了，无法区分。
      把基准率一并输出，才能暴露这个混淆。
    """
    if not relevants or corpus_size <= 0:
        return 0.0
    return (sum(len(r) for r in relevants) / len(relevants)) / corpus_size


def add_lift(metrics: dict, prefix: str, base: float,
             k_values: tuple[int, ...] = DEFAULT_K_VALUES) -> dict:
    """把 precision@k 归一化成 lift@k = precision@k / 基准率

    lift 的含义：比"随机取一个片段"的相关概率高多少倍。

    ★ 它解决什么、不解决什么（别指望它超出这个范围）：
      - 解决：**跨分组（跨分块配置）的可比性**。基准率不同的两组，lift 可以横向比。
      - 不解决：**标注本身的循环性**。片段级正例是用嵌入相似度定义的，
        因此在某个检索器与标注生成器同源时，lift 仍会偏向它。
        循环性只能靠"换一种与嵌入无关的标注"来对抗（见 caliber_text("kw")）。
    """
    if base <= 0:
        return metrics
    for k in k_values:
        key = f"{prefix}_precision@{k}"
        if key in metrics:
            metrics[f"{prefix}_lift@{k}"] = round(metrics[key] / base, 3)
    return metrics


def metric_names(k_values: tuple[int, ...] = DEFAULT_K_VALUES, prefix: str = "") -> list[str]:
    """指标名顺序（供报告统一渲染）"""
    tag = f"{prefix}_" if prefix else ""
    names = [f"{tag}hit_rate@{k}" for k in k_values] + [f"{tag}recall@{k}" for k in k_values]
    names += [f"{tag}precision@{k}" for k in k_values]
    names += [f"{tag}lift@{k}" for k in k_values]
    names += [f"{tag}mrr", f"{tag}ndcg@{max(k_values)}"]
    return names


def dedupe_keep_order(ids: list[str]) -> list[str]:
    """文档级排序去重（保留首次出现的排名）

    为什么不直接留着重复：同一份文档的多个片段会占据 top-k 的多个位置，
    不去重会把 hit/mrr/nDCG 都算虚高 —— 文档级评测里"同一份文档命中两次"没有意义。
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in ids:
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def caliber_text(label_level: str) -> str:
    """口径说明文本（写进报告，避免只看表头误读）"""
    if label_level == "doc":
        return ("文档级标注：每个 query 的正例 = 评测集的 source 字段（应命中的 .txt）。"
                "规则确定、可复现；局限是文档数少时指标容易饱和。")
    if label_level == "kw":
        return ("片段级**对抗性口径**：正例 = 片段文本里出现了该问题的 answer_keywords（纯词面匹配）。"
                "★ 这个口径**系统性偏袒 BM25 这类关键词检索**、对向量检索不利 —— 这是故意的："
                "若向量检索在此口径下仍然胜出，结论就比弱标注口径硬；若反转，则说明"
                "弱标注口径下的优势确实来自'标注生成器与向量检索同源'的循环性。两种结果都有信息量。")
    return ("片段级**弱标注**：正例 = 与标准答案嵌入相似度 ≥ 阈值的片段。"
            "★ 这是自动生成的弱标注，不是人工标注；且它由嵌入模型生成，"
            "与该口径下的向量检索**同源**，存在循环性（详见报告的口径说明）。")
