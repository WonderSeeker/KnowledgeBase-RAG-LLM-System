# RAG 评测报告

- **rerankon_v2_local_c300_o30_hybrid**：rerankon_v2_local_c300_o30_hybrid | 模式=ir 裁判=deterministic | 检索=hybrid top_k=4 候选=10 重排=gte-rerank-v2 | 语料分块=300/30 ×3 次均值（实测噪声地板 1.0 题 / 36） | 2026-10-13 15:30:07

> **[警告]** IR 指标（确定性，非 RAGAS）：测检索排序质量，不含生成质量。doc_* 文档级（无循环）/ kw_* 对抗口径（无嵌入）/ chunk_* 弱标注（嵌入同源）。本报告为 3 个独立进程的均值，实测噪声地板 0.0278（1.0 题 / 36）。

## 指标对比

| 指标 | rerankon_v2_local_c300_o30_hybrid |
|---|---|
| doc_hit_rate@1<br>文档级命中率@1 | 0.7593 |
| doc_hit_rate@3<br>文档级命中率@3 | 0.9074 |
| doc_precision@1<br>文档级精度(噪声)@1 | 0.7593 |
| doc_precision@3<br>文档级精度(噪声)@3 | 0.3025 |
| doc_precision@5<br>文档级精度(噪声)@5 | 0.1815 |
| doc_recall@1<br>文档级召回率@1 | 0.7593 |
| doc_recall@3<br>文档级召回率@3 | 0.9074 |
| doc_recall@5<br>文档级召回率@5 | 0.9074 |
| doc_mrr<br>文档级MRR | 0.8333 |
| doc_ndcg@5<br>文档级nDCG@5 | 0.8527 |
| kw_hit_rate@1<br>对抗口径命中率@1 | 0.7593 |
| kw_hit_rate@3<br>对抗口径命中率@3 | 0.8241 |
| kw_precision@1<br>对抗口径精度(噪声)@1 | 0.7593 |
| kw_precision@3<br>对抗口径精度(噪声)@3 | 0.6204 |
| kw_precision@5<br>对抗口径精度(噪声)@5 | 0.4870 |
| kw_lift@1<br>对抗口径lift(相对基准率)@1 | 7.4670 |
| kw_lift@3<br>对抗口径lift(相对基准率)@3 | 6.1010 |
| kw_lift@5<br>对抗口径lift(相对基准率)@5 | 4.7897 |
| kw_recall@1<br>对抗口径召回率@1 | 0.0266 |
| kw_recall@3<br>对抗口径召回率@3 | 0.0594 |
| kw_recall@5<br>对抗口径召回率@5 | 0.0729 |
| kw_mrr<br>对抗口径MRR | 0.7986 |
| kw_ndcg@5<br>对抗口径nDCG@5 | 0.5558 |
| chunk_hit_rate@1<br>片段级(弱标注)命中率@1 | 0.7963 |
| chunk_hit_rate@3<br>片段级(弱标注)命中率@3 | 0.8704 |
| chunk_precision@1<br>片段级(弱标注)精度(噪声)@1 | 0.7963 |
| chunk_precision@3<br>片段级(弱标注)精度(噪声)@3 | 0.6605 |
| chunk_precision@5<br>片段级(弱标注)精度(噪声)@5 | 0.4981 |
| chunk_lift@1<br>片段级(弱标注)lift(相对基准率)@1 | 7.9010 |
| chunk_lift@3<br>片段级(弱标注)lift(相对基准率)@3 | 6.5530 |
| chunk_lift@5<br>片段级(弱标注)lift(相对基准率)@5 | 4.9423 |
| chunk_recall@1<br>片段级(弱标注)召回率@1 | 0.0419 |
| chunk_recall@3<br>片段级(弱标注)召回率@3 | 0.0656 |
| chunk_recall@5<br>片段级(弱标注)召回率@5 | 0.0748 |
| chunk_mrr<br>片段级(弱标注)MRR | 0.8379 |
| chunk_ndcg@5<br>片段级(弱标注)nDCG@5 | 0.5859 |
| chunk_hit_rate@5<br>片段级(弱标注)命中率@5 | 0.9074 |
| doc_hit_rate@5<br>文档级命中率@5 | 0.9074 |
| kw_hit_rate@5<br>对抗口径命中率@5 | 0.8518 |

## 指标口径说明

**rerankon_v2_local_c300_o30_hybrid —— 口径：确定性 IR 指标（**不需要 LLM 裁判**）**

| 指标 | 本口径下的实际算法 |
|---|---|
| hit_rate | top-k 里是否至少命中一个正例（0/1，对 k 个 query 取平均） |
| recall | top-k 覆盖了正例集合的多大比例（注意：会被语料粒度污染，跨分块配置不可比） |
| precision | top-k 里有多大比例是正例（噪声视角；与语料总片段数无关，可跨分块配置比较） |
| mrr | 第一个正例的排名倒数（1/rank），对「排得够不够前」最敏感 |
| ndcg | 二元相关性的 nDCG@k；正例全部排在最前时为 1.0 |

- **rerankon_v2_local_c300_o30_hybrid / `doc_*`**：文档级标注：每个 query 的正例 = 评测集的 source 字段（应命中的 .txt）。规则确定、可复现；局限是文档数少时指标容易饱和。
- **rerankon_v2_local_c300_o30_hybrid / `chunk_*`**：片段级**弱标注**：正例 = 与标准答案嵌入相似度 ≥ 阈值的片段。★ 这是自动生成的弱标注，不是人工标注；且它由嵌入模型生成，与该口径下的向量检索**同源**，存在循环性（详见报告的口径说明）。；相似度阈值 0.6，语料 872 个片段，平均每 query 87.89 个正例
- **rerankon_v2_local_c300_o30_hybrid / `kw_*`**：片段级**对抗性口径**：正例 = 片段文本里出现了该问题的 answer_keywords（纯词面匹配）。★ 这个口径**系统性偏袒 BM25 这类关键词检索**、对向量检索不利 —— 这是故意的：若向量检索在此口径下仍然胜出，结论就比弱标注口径硬；若反转，则说明弱标注口径下的优势确实来自'标注生成器与向量检索同源'的循环性。两种结果都有信息量。

> 只测**检索排序质量**，不含生成质量。`doc_*` 为文档级主口径；`chunk_*` 为片段级**弱标注**口径（自动生成，非人工标注，阈值见报告 caliber）。

## 运行开销

| 项 | rerankon_v2_local_c300_o30_hybrid |
|---|---|
| 平均检索片段数 | 4.0 |
| 平均单题耗时(s) | None |
| 整轮评测耗时(s) | 10.35 |
