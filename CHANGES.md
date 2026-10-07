# 改造记录（CHANGES）

> 记录本仓库的改造过程：原项目是什么 / 我改了什么 / 量化结果。
> 每个阶段一条，量化数据全部来自本机实测。
>
> 先读这一条：本文件里标注「离线启发式」的数字表（4.3 检索策略对比、4.4 分块参数对比）
> **都不是结论** —— mock 模式下检索腿与打分器都是替身。哪些能当结论、哪些必须等真实嵌入复测，
> 见文末「**六、结论有效性划定**」。写 README / 简历前请务必先看那一节。

- 上游项目：`lhh737/KnowledgeBase-RAG-LLM-System`（MIT）
- 本地路径：`D:\Projects\KnowledgeBase-RAG-LLM-System`
- 运行环境：`.venv` + **Python 3.11.9**（不可升级到 3.12，见文末「禁止事项」）
- 技术栈：Streamlit / LangChain 0.1.x / Chroma 0.4.24 / DashScope（通义千问）

---

## 一、原项目是什么

一个 demo 级的本地知识库 RAG 问答系统，共 6 个核心 py 文件：

| 文件 | 职责 |
|---|---|
| `config_data.py` | 模型名、Chroma 路径、分块参数、检索 k |
| `knowledge_base.py` | 文档读取、切分、MD5 去重、写入向量库 |
| `vector_stores.py` | Chroma 检索器封装 |
| `rag.py` | 检索 + 生成链路编排（LCEL） |
| `app_chat.py` | Streamlit 问答界面（流式输出） |
| `app_upload.py` | Streamlit 上传界面（仅 txt） |

**原始数据流**

```
上传：app_upload.py → KnowledgeBaseService.upload_by_str
      → MD5 去重 → RecursiveCharacterTextSplitter 切分
      → DashScopeEmbeddings 向量化 → Chroma(./chroma_db)

问答：app_chat.py → RagService.chain
      → retriever(k=1) → format_document 拼 context
      → ChatPromptTemplate → ChatTongyi(qwen3-max) → StrOutputParser
      → RunnableWithMessageHistory(./chat_history/<session_id>)
```

原项目缺失：**没有任何评测环节**，效果好坏全靠肉眼看；没有 API；没有埋点。

---

## 二、阶段 1：跑通基线（本次改造）

### 2.1 修复的 6 个问题

| # | 问题 | 位置（改前） | 影响 | 修法 |
|---|---|---|---|---|
| 1 | `.env` 形同虚设 | 全项目无 `load_dotenv()` | 建了 `.env` 也读不到 key，只能靠系统环境变量 | `config_data.py` 顶部 `load_dotenv(BASE_DIR/".env")` |
| 2 | 路径依赖工作目录 | `"./chroma_db"`、`"./md5.text"` | 从不同目录启动会读写不同的库（README 的 Q1 坑） | 改为基于 `__file__` 的绝对路径 |
| 3 | `.gitignore` 漏项 | 缺少 `.env`、`chroma_db/`、`md5.text` | 密钥与向量库会被 commit | 补齐忽略项 |
| 4 | 上传只支持 txt | `app_upload.py` `type=['txt']` | 无法上传 PDF | 增加 pdf 分支，`knowledge_base.extract_text()` 用 pypdf 抽页 |
| 5 | 检索 k 命名错误 | `similarity_threshold = 1` 被当 k 用 | 语义混乱，容易误改 | 新增 `retrieve_k`，旧名保留兼容（值仍为 1，**行为不变**） |
| 6 | 嵌入模型两处不一致 | `knowledge_base.py` 硬编码 `text-embedding-v4` | 改配置会导致入库/检索模型不同，检索失效 | 统一走 `config.embedding_model_name` |

### 2.2 新增的能力

| 新增 | 作用 |
|---|---|
| `.env`（不进 git） | 密钥只落本地 |
| `scripts/ingest.py` | 命令行批量入库 txt/pdf，**绕开浏览器**，让入库可自动化、可复现 |
| `RagService.retrieve(question)` | 单独执行检索，返回命中片段（阶段 2 的 RAGAS 需要 contexts） |
| `RagService.answer(question)` | 一次性问答，返回答案 + 命中片段 + 分段耗时，供脚本与评测调用 |
| 埋点 | 入库片段数/耗时、检索命中片段数、端到端耗时（打进 `_logs\rag.out` 与页面） |

### 2.3 基线指标（待填 —— 需要真实 API Key 后跑一次）

| 指标 | 基线值 | 采集方式 |
|---|---|---|
| 检索命中片段数 | 1 | `config.retrieve_k = 1` |
| 分块参数 | chunk_size=1000 / overlap=100 / 阈值 `max_spliter_char_number=1000` | `config_data.py` |
| 嵌入模型 | text-embedding-v4 | `config_data.py` |
| 生成模型 | qwen3-max（ChatTongyi） | `config_data.py` |
| 语料规模 | 3 个 txt（assets 自带） | `scripts/ingest.py` 输出 |
| 端到端耗时 | 待实测 | `[问答] 端到端耗时 X s` |
| 单文件入库耗时 | 待实测 | `[入库] 向量化+写库耗时 X s` |

> 填表命令：`cd D:\Projects; .\svc.ps1 start rag` → 提问 → 看 `_logs\rag.out`；验完 `.\svc.ps1 stop rag`

### 2.4 已知的基线缺陷（阶段 3 的优化靶子）

- `retrieve_k = 1`：只喂 1 个片段给大模型，`context_recall` 天花板极低。
- 纯向量检索：没有 BM25 关键词召回，专有名词/型号类问题容易漏。
- 没有 Rerank：召回结果无二次排序。
- 切分阈值语义混乱：`len(data) > 1000` 才切分，<1000 字符的文档整篇入库（两份示例文本都 <1000 字符）。

---

## 三、阶段 2：接入 RAGAS 评测层

### 3.1 依赖预研（本次改造最大的技术风险，已实测排除）

项目的 `langchain-core` 锁在 0.1.53，而 **ragas 0.2+ 要求 `langchain-core>=0.2`**，
一旦装上会连带升级 pydantic / numpy，项目当场跑不起来（numpy 必须 <2）。

处理过程与结论：

| 动作 | 结果 |
|---|---|
| `pip install --dry-run ragas==0.1.21` | 只新增包，**不动 langchain / pydantic / numpy** |
| 实际安装（带约束文件） | 新增 ragas 0.1.21、datasets、openai、langchain-openai 等 13 个包 |
| 装完核对版本 | langchain 0.1.20、core 0.1.53、community 0.0.38、chroma 0.1.4、chromadb 0.4.24、chroma-hnswlib 0.7.3、numpy 1.26.4、pydantic 2.13.5、streamlit 1.39.1 —— **全部与改造前一致** |
| 唯一变化 | `fsspec` 2026.9.0 → 2026.6.0（datasets 要求；项目本身不用 fsspec） |
| 4 个指标 API 导入 | `ragas 0.1.21` + `datasets 5.0.1` 下全部导入成功 |

新增 `evaluation/constraints-eval.txt` 把主依赖钉死，防止以后再装评测依赖时被顺带升级。

### 3.2 新增文件

| 文件 | 作用 |
|---|---|
| `evaluation/dataset.py` | 评测集构建：29 条内置样本 + JSONL 读写（`--init` 可重新生成） |
| `evaluation/eval_set.jsonl` | **权威评测集**，一行一条，可直接手工增删，git diff 友好 |
| `evaluation/offline.py` | 离线替身：本地切分 + 字面检索 + 启发式代理指标，**不需要 API Key** |
| `evaluation/run_ragas.py` | 跑评测主入口：RAG 链路产出答案/片段 → RAGAS 打分 → 落 JSON |
| `evaluation/report.py` | 出报告：单份指标表 / 两份「改造前 vs 改造后」对比表（Markdown） |
| `evaluation/requirements-eval.txt` | 评测层依赖（不改动主 requirements.txt） |
| `evaluation/constraints-eval.txt` | 主依赖版本约束，防误升级 |

### 3.3 评测集构成（29 条，全部出自 `assets/` 语料）

- 按语料：`尺码推荐.txt` 5 条、`洗涤养护.txt` 15 条、`颜色推荐.txt` 9 条
- 按类型：单点事实 12 条、精确数值 8 条、跨段落归纳 7 条、否定性提问 2 条

按类型分布是刻意设计的：**「跨段落归纳」和「精确数值」最容易因 `top_k` 过小而漏检**，
阶段 3 的优化效果会主要从这两类里体现出来。

### 3.4 一条命令跑出报告

```powershell
cd D:\Projects\KnowledgeBase-RAG-LLM-System

# 真实 RAGAS 评测（需要 .env 里的 DASHSCOPE_API_KEY）
.\.venv\Scripts\python.exe evaluation\run_ragas.py --mode real --name baseline

# 离线自检（不需要 Key，验证流程可跑通，不烧 token）
.\.venv\Scripts\python.exe evaluation\run_ragas.py --mode mock

# 出报告 / 出对比表
.\.venv\Scripts\python.exe evaluation\report.py evaluation\report_baseline.json
.\.venv\Scripts\python.exe evaluation\report.py evaluation\report_baseline.json evaluation\report_optimized.json --out evaluation\REPORT.md

# 只重新打分，不重跑模型（换指标/换裁判时省钱）
.\.venv\Scripts\python.exe evaluation\run_ragas.py --reuse evaluation\runs\baseline_real.jsonl
```

裁判模型设计（`--judge-provider`）：

- `dashscope`（默认）：走 DashScope 的 **OpenAI 兼容端点** `https://dashscope.aliyuncs.com/compatible-mode/v1`，
  这是 RAGAS 最稳的接法；裁判模型默认 `qwen-plus`（打分模型要快，别用 `qwen3-max`）
- `dashscope-native`：`LangchainLLMWrapper(ChatTongyi(...))`
- `ollama`：本地 `http://127.0.0.1:11434/v1`，阶段 4 做「零成本」对比时用

### 3.5 已实测的数据（⚠️ 离线启发式，**不是**真实效果）

用同一套评测集，只改 `top_k`（1 → 4），验证「检索一变、指标就跟着变」这条因果链是通的：

| 指标 | top_k=1 | top_k=4 | Δ | 变化 |
|---|---|---|---|---|
| faithfulness（忠实度） | 1.0000 | 1.0000 | +0.0000 | +0.0% |
| answer_relevancy（答案相关性） | 0.2146 | 0.2146 | +0.0000 | +0.0% |
| context_recall（上下文召回） | 0.6972 | **0.7547** | **+0.0575** | **+8.2%** |
| context_precision（上下文精度） | 0.6972 | 0.3256 | -0.3716 | -53.3% |
| 平均检索片段数 | 1.0 | 4.0 | - | - |

说明：faithfulness 恒为 1.0 是因为离线替身把片段原样当答案（不会幻觉），
真实模型下这个数会降下来；`context_precision` 随 k 增大而下降，符合"召回了但噪声变多"的直觉。
**这些数字只能证明流水线正确，不能当作改造效果。**

### 3.6 阶段 2 验收状态

- [x] 一条命令跑出评测报告，报告含 4 个指标（`report_baseline.json` 结构已就绪）
- [x] 评测集 29 条（20~30 条要求达成）
- [x] 4 个指标全部接入 RAGAS（`faithfulness` / `answer_relevancy` / `context_recall` / `context_precision`）
- [x] 对比表功能已就绪（阶段 3 直接复用）
- [ ] **真实 RAGAS 分数** —— 阻塞在 `DASHSCOPE_API_KEY`，填入 `.env` 后执行 3.4 的第一条命令即可产出

### 3.7 踩到的坑（记录备查）

1. **Windows 控制台是 GBK**：脚本里用 `print("⚠️ ...")` 直接抛 `UnicodeEncodeError`。
   已全部改为纯文本 `[警告]`。文件本身仍用 `encoding="utf-8"` 写入，不受影响。
2. **PowerShell 显示 UTF-8 中文会乱码**：`Get-Content eval_set.jsonl` 看到的是乱码，
   但用 Python 以 utf-8 读取完全正常——是 PS 5.1 的显示编码问题，文件本身没问题。
3. `pip` 装含中文注释的 `requirements.txt` 会 `UnicodeDecodeError: 'gbk'`，
   需先 `$env:PYTHONUTF8='1'`（本次安装已用到）。

---

## 四、阶段 3：检索优化 + A/B 对比

### 4.1 新增内容

| 新增 | 作用 |
|---|---|
| `retrievers.py` | 检索策略层：向量 / BM25 / RRF 混合 / DashScope 重排，统一 `invoke()` 契约 |
| `requirements-optim.txt` | `rank-bm25==0.2.2` + `jieba==0.42.1`（实测只新增 2 个包，主依赖零变动） |
| `evaluation/experiments.py` | 对比实验驱动：入库 → 评测 → 汇总成一张横向表，一条命令 |
| `scripts/ingest.py`（升级） | 支持 `--chunk-size/--chunk-overlap/--collection/--rebuild/--dry-run` |
| `evaluation/run_ragas.py`（升级） | 新增 `--retriever/--candidate-k/--rerank/--collection/--chunk-size/--chunk-overlap` |

`config_data.py` 新增参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `retrieval_method` | `vector` | `vector` / `bm25` / `hybrid` |
| `candidate_k` | 10 | 粗排候选数（混合与重排都从这里精选） |
| `rrf_k` | 60 | RRF 融合常数 |
| `rerank_enabled` | `False` | 是否开启重排 |
| `rerank_model` | `gte-rerank-v2` | DashScope 重排模型 |

### 4.2 三个关键技术选择及理由

1. **混合检索用 RRF 而不是加权求和**
   BM25 分值与余弦相似度量纲完全不同，直接加权要调参且不稳定；RRF 只看排名不看分值，天然免调参。
2. **BM25 语料直接从 Chroma 集合读**
   保证两路检索面对的是**同一份语料、同一套分块**，避免"两路各用一套索引"这种隐蔽 bug。
3. **重排失败必须降级而不是崩**
   重排依赖外部 API，网络/额度任何一环抖动都不该让问答挂掉。当前实现失败时退回粗排顺序并打警告，
   无 Key 时同样降级（已实测）。

### 4.3 检索策略对比（⚠️ 离线启发式，不是 RAGAS）—— **k 已对齐**

> ⚠️ **本表是替身数据，不可作为结论。** 判定理由见 6.2。
> 保留它的唯一价值：证明同一套 k 口径下能出多路对比表，且 k 不一致会被拦截。
> → **真实重测已完成，见下方 4.3-R**（local 512 维真实嵌入 + 确定性 IR 指标）。

**先纠正一版错误结论。** 阶段 3 第一版的表里 vector 用 `k=1`、bm25/hybrid 用 `k=4`，
于是我写下了"混合检索召回 +7.3%"。这个对比是无效的：k 不同时，检索片段数不同，
`context_precision` / `context_recall` 天然不可比（返回 4 条当然比返回 1 条噪声多）。

补上 `vector(k=4)` 对照、三路统一在 **k=4** 后，真实情况是：

| 指标（29 条评测集，k 全部 = 4，候选 12） | vector | bm25 | hybrid |
|---|---|---|---|
| faithfulness | 0.9655 | 0.9310 | 0.9655 |
| answer_relevancy | 0.2146 | 0.2105 | 0.2146 |
| context_recall | 0.7468 | 0.7159 | **0.7481** |
| context_precision | 0.4610 | **0.4793** | 0.4313 |

**结论修正**：

1. 之前那个 **+7.3% 主要来自 `k` 从 1 提到 4，不是混合检索的功劳**。
   单看 vector：`k=1` 召回 0.6972 → `k=4` 召回 0.7468（+7.1%）。
2. **k 对齐后，三种策略差异很小**（召回 0.7159~0.7481，极差 3.2%），
   hybrid 的 precision 反而最低（0.4313 vs vector 0.4610，-6.4%）。
3. 原因很明确：**离线模式的"向量腿"本身就是字面检索替身**，
   于是 hybrid = 字面 + 字面，两路同源，RRF 融合自然拿不到额外增益。
   这是评测替身的固有局限，不是 BM25/RRF 实现有问题。

> **所以：混合检索与重排的真实增益，必须在 `--mode real`（真向量检索）下才能测出来。**
> 离线表只能证明「流程正确」和「指标对 k 敏感」这两件事。

为了以后不再犯同样的错，`report.py` 现在会**主动拦截** k 不一致的对比：
各报告 `top_k` 不同时，终端与 Markdown 里都会打印口径不一致的警告。

### 4.3-R 真实重测：检索策略对比（local 嵌入 512 维 + 确定性 IR 指标）★有效结论

命令（三路统一 `k=4`，语料 6 片段，集合 `rag_local_c1000_o100`）：

```powershell
.\.venv\Scripts\python.exe evaluation\run_ir.py --name ir_retriever ^
    --configs 1000/100 --retrievers vector,bm25,hybrid --top-k 4 --candidate-k 10 --embedding-backend local
```

**文档级指标（主口径：正例 = 评测集 `source`，人工出处、非循环）**

| 指标 | vector | bm25 | hybrid |
|---|---|---|---|
| doc_hit_rate@1 | 1.0000 | **0.8966** | 1.0000 |
| doc_mrr | 1.0000 | 0.9138 | 1.0000 |
| doc_ndcg@5 | 1.0000 | 0.9183 | 1.0000 |
| n（样本数） | 29 | 29 | 29 |

三个不同设置下 BM25 都稳定落后（换算成题数：`0.8966×29=26`，`0.9655×29=28`）：

| 设置 | vector | bm25 | hybrid | BM25 落后题数 |
|---|---|---|---|---|
| 1000/100，k=4 | 1.0000 | 0.8966 | 1.0000 | **3 题**（26/29） |
| 300/30，k=4 | 1.0000 | 0.9655 | 1.0000 | 1 题（28/29） |
| 150/15，k=1 | 1.0000 | 0.9655 | 1.0000 | 1 题（28/29） |

**片段级指标（弱标注口径，仅 1000/100 + k=4）**

| 指标 | vector | bm25 | hybrid |
|---|---|---|---|
| chunk_mrr | 0.9483 | 0.8391 | 0.9598 |
| chunk_ndcg@5 | 0.9596 | 0.7328 | 0.9565 |
| chunk_recall@3 | 0.9770 | 0.6839 | 0.9540 |

**结论（连同必须一起说的局限）**

1. ✅ **可复现**：`vector` 与 `hybrid` 在文档级打平且都到 1.0，**BM25 稳定落后**（三个设置一致）。
2. ✅ **落后原因可定位**：集中在语义改写类问题 —— 「怎样穿显得更瘦？」在 5.7 已实测
   向量命中、BM25 **0 命中**。
3. ⚠️ **幅度小、且在噪声内**：差 1~3 题（3.4% ~ 10.3%），n=29，**不足以宣称统计显著**。
   只能说"方向一致、且与样本级证据吻合"。要下强结论需要更多文档、更多 query。
4. ⚠️ **片段级差距不能当公平对比**：`chunk_*` 的正例是「与标准答案嵌入相似度 ≥ 阈值的片段」，
   这个口径**结构性地偏袒嵌入型检索器**（用嵌入定义相关性，再测嵌入检索）。
   vector / hybrid 大幅领先是**预期内的口径偏袒**，不是独立证据。
5. ⚠️ **文档级在此语料上近乎饱和**：只有 3 份文档，top-4 几乎必然覆盖正确文档；
   `doc_precision@k` 因此退化（只有 3 个文档，`precision@5` 恒为 0.2），不可解读。

### 4.4 分块参数对比实验（⚠️ 离线启发式，不是 RAGAS）

> ⚠️ **本表是替身数据，不可作为结论。** mock 模式下「检索」与「指标」**都是替身**，
> 两个替身相乘，结论没有可迁移性。判定理由见 6.2。表里唯一的客观事实是**片段数那一行**。
> → **真实重测已完成，见下方 4.4-R**。

`evaluation/experiments.py --mode mock --name chunk --configs 150/15,300/30,600/60,1000/100 --retriever hybrid --top-k 4 --candidate-k 32`

（四组统一 `top_k=4` / `candidate_k=32`，只改分块参数）

| 指标 | 150/15 | 300/30 | 600/60 | 1000/100 |
|---|---|---|---|---|
| 语料片段总数（`--dry-run` 实测） | 32 | 15 | 9 | 6 |
| context_recall | 0.6633 | 0.7210 | 0.7439 | **0.7481** |
| context_precision | 0.2835 | 0.3296 | 0.3774 | **0.4313** |
| answer_relevancy | **0.3409** | 0.2875 | 0.2501 | 0.2146 |
| faithfulness | 0.9655 | 0.9655 | 0.9655 | 0.9655 |

趋势单调、方向合理：**分块越小 → 片段越聚焦、答案相关性越高，但召回与精度下降；
分块越大 → 相反**。这是一条真实的权衡曲线，不是"某组参数一定最好"。

其中 300/30 与 600/60 两组数值与修复分块缺陷**之前**完全一致——这是预期内的回归验证：
这两个配置的 `chunk_size` 都 ≥「尺码推荐.txt」的 270 字，修复前后该文件都是 1 个片段。
真正体现差异的是新增的 150/15（修复前它同样只有 1 个片段，修复后是 2 个，详见 4.8）。

### 4.4-R 真实重测：分块参数对比（local 嵌入 + IR 指标）★有效结论

命令（固定检索策略 `hybrid`、固定 `k=4`，只变分块参数）：

```powershell
.\.venv\Scripts\python.exe evaluation\run_ir.py --name ir_chunk ^
    --configs 150/15,300/30,600/60,1000/100 --retriever hybrid --top-k 4 --candidate-k 32 --embedding-backend local
```

**文档级指标：四个配置全部顶到 1.0，没有区分度**

| | 150/15 | 300/30 | 600/60 | 1000/100 |
|---|---|---|---|---|
| 语料片段数 | 32 | 15 | 9 | 6 |
| doc_hit_rate@1 / doc_mrr / doc_ndcg@5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**片段级指标（弱标注）：这里出现了真实的权衡**

| 指标 | 150/15 | 300/30 | 600/60 | 1000/100 | 形态 |
|---|---|---|---|---|---|
| chunk_precision@3 | **0.9080** | 0.7701 | 0.7011 | 0.6092 | 单调 |
| chunk_precision@5 | **0.7172** | 0.5931 | 0.4897 | 0.3862 | 单调 |
| chunk_recall@3 | 0.2992 | 0.5967 | 0.7678 | **0.9540** | 单调，但**不可解读**（见下） |
| chunk_mrr | **0.9828** | 0.9598 | 0.9052 | 0.9598 | 非单调 |
| chunk_ndcg@5 | 0.8372 | 0.8602 | 0.8517 | **0.9565** | 非单调 |

**结论（含一个必须点明的指标陷阱）**

1. ✅ **precision 随分块变小而单调变好**：150/15 的 `chunk_precision@3` 是 1000/100 的 1.49 倍
   （0.9080 vs 0.6092）—— 分块越细，喂进 top-k 的噪声越少。这是可复现的结论。
2. ⚠️ **`recall@k` 在跨分块配置时不可解读**：它随分块变大而"变好"（0.2992 → 0.9540）
   **不是检索变好了**，而是语料变小了 —— 6 个片段时 top-4 已经覆盖 67% 的语料。
   ★ 这正是要同时输出 `precision@k` 的原因：它只问"top-k 里有多少是相关的"，与语料总片段数无关。
   **不要把 recall 那一行读成"1000/100 最好"。**
3. ⚠️ **没有单一赢家**：`ndcg@5` 反而偏爱 1000/100（0.9565），`mrr` 偏爱 150/15（0.9828）——
   这本身就是"分块是权衡、不存在最优解"的证据。
4. ⚠️ **样本规模不足以定论**：3 份文档、29 条 query、语料 6~32 个片段。`doc_*` 全部饱和，
   `chunk_*` 又有弱标注偏袒，所以上面只能支撑"precision 随细粒度变好"这一条窄结论。

### 4.5 这一步挖出来的问题

1. **`max_spliter_char_number = 1000` 让"调小 chunk_size"对短文档完全无效** —— 已修复，见 4.8。
2. **语义改写类问题是字面检索的死角**
   `color-06「怎样穿显得更瘦？」` 在离线替身下 **0 命中**——语料里写的是"显瘦搭配"，
   字面/BM25 都匹配不上，只有真向量检索能救。这条样本正好是验证向量检索价值的标杆。
3. **基线的 `top_k=1` 有 3% 的问题直接 0 命中**（平均片段数 0.97），等于大模型收到"无相关参考资料"。
4. **对比实验的口径必须显式校验** —— 已由 `report.py` 的 k 一致性检查兜住（见 4.3）。
5. **Chroma 遥测报错刷 stderr** —— 根因不是"遥测没关"，而是 **chromadb 0.4.24 的 posthog 适配层
   用 3 个位置参数调 `posthog.capture`，新版 posthog 只接受 1 个**，异常发生在签名校验阶段，
   所以 `Settings(anonymized_telemetry=False)`（只置 `posthog.disabled`）挡不住这行 `logger.error`。
   现改为定向把 `chromadb.telemetry.product.posthog` 这个 logger 提到 CRITICAL：
   既不发数据、也不再刷日志（已用真实入库链路验证无该输出）。

### 4.6 命令

```powershell
cd D:\Projects\KnowledgeBase-RAG-LLM-System

# 不联网看清分块效果（不需要 Key）
.\.venv\Scripts\python.exe scripts\ingest.py --dry-run --chunk-size 300 --chunk-overlap 30

# 离线跑通实验流程（分块真实生效，检索与打分用离线替身）
.\.venv\Scripts\python.exe evaluation\experiments.py --mode mock --name chunk ^
    --configs 150/15,300/30,600/60,1000/100 --retriever hybrid --top-k 4 --candidate-k 32

# 检索策略对比：★ 必须显式固定 --top-k，否则 context_precision 不可比
.\.venv\Scripts\python.exe evaluation\experiments.py --mode mock --name alignk4 ^
    --configs 1000/100 --retrievers vector,bm25,hybrid --top-k 4 --candidate-k 12

# 真实实验（需要 Key）：每组分块参数入独立集合，再用 RAGAS 打分
.\.venv\Scripts\python.exe evaluation\experiments.py --mode real --name chunk ^
    --configs 500/50,1000/100,1500/200 --retriever hybrid --top-k 4 --candidate-k 10 --rerank

# 基线 vs 优化 的真实对比表
.\.venv\Scripts\python.exe evaluation\report.py evaluation\report_baseline.json evaluation\report_optimized.json --out evaluation\REPORT.md
```

### 4.7 阶段 3 验收状态

- [x] BM25 + 向量混合检索（RRF 融合）
- [x] Rerank 接入（DashScope `gte-rerank-v2`），失败降级已实测
- [x] 分块大小/重叠度对比实验流程（`--dry-run` 试算 + 多组评测 + 横向表）
- [x] 「改造前 vs 改造后」对比表能力（`report.py` 支持 1/N 份报告，且会拦截 k 不一致的对比）
- [x] 对比实验的 k 已对齐（4.3 的修正表）
- [x] `chunk_size` 对短文档失效的缺陷已修复（4.8）
- [ ] **真实 `report_optimized.json` 与真实对比表** —— 阻塞在 `DASHSCOPE_API_KEY`
- [ ] 混合检索 / 重排的真实收益数字 —— 必须 `--mode real` 才能测（离线替身下两路同源，见 4.3）

---

### 4.8 缺陷修复记录：`chunk_size` 对短文档完全失效

**缺陷**（原项目遗留，非本次改造引入）

`knowledge_base.upload_by_str` 用 `max_spliter_char_number(=1000)` 做**硬门槛**：

```python
if len(data) > config.max_spliter_char_number:
    knowledge_chunks = self.spliter.split_text(data)
else:
    knowledge_chunks = [data]          # ← ≤1000 字的文档一律不切分
```

后果：「尺码推荐.txt」只有 270 字，**无论 `chunk_size` 设 300 还是 1500，永远是 1 个片段**。
分块参数对短文档彻底失效——而这份尺码表恰恰是最需要精确检索（只想召回相关那几行）的内容。
这个门槛在本轮分块对比实验里第一次被暴露出来（见 4.4 的 `--dry-run` 实测）。

**修复**

判定统一交给 splitter：文本短于 `chunk_size` 时它自己就会返回 1 个片段，
文本更长时按 `chunk_size`/`chunk_overlap` 切分。语义单一、`chunk_size` 真实生效。
`max_spliter_char_number` 保留为配置项（兼容旧代码），但**不再参与任何判定**，
三处判定（`knowledge_base.py` / `evaluation/offline.py` / `scripts/ingest.py --dry-run`）同步收敛。

**证据 1：切分行为前后对照（`chunk_size=150, chunk_overlap=15`）**

| | 片段数 | 各片段长度 |
|---|---|---|
| 修复前 | **1** | 270（被 1000 的硬门槛挡掉，`chunk_size` 失效） |
| 修复后 | **2** | 133, 135 |

**证据 2：真实入库链路端到端验证**（用 `FakeEmbeddings` 绕开 API Key，但走完整的
`KnowledgeBaseService.upload_by_str` → 切分 → 写 Chroma → MD5 去重）

```
[入库] 文件=尺码推荐.txt 原文长度=270 切分片段数=2 (chunk_size=150 chunk_overlap=15 集合=probe150b)
返回 = [Success]内容已经成功载入向量库（2 个片段，耗时 0.00s）
集合内片段数 = 2
各片段长度 = [135, 133]
重复入库 = [Repeat] 内容已存在知识库          ← 去重逻辑未被破坏
```

**证据 3：`--dry-run` 全量扫描（修复后）**

| chunk_size/overlap | 尺码推荐.txt | 洗涤养护.txt | 颜色推荐.txt | 合计 |
|---|---|---|---|---|
| 150/15 | **2** | 19 | 11 | 32 |
| 300/30 | 1（270 < 300，理应 1 个） | 9 | 5 | 15 |
| 1500/200 | 1 | 2 | 1 | 4 |

> 注：`300/30` 下「尺码推荐.txt」仍是 1 个片段是**正确行为**——原文 270 字短于 chunk_size=300。
> 修复要解决的是"150/15 也切不出来"，而不是"任何配置都必须切出多段"。

**对分块实验的影响**：新增了 `150/15` 这一组（修复前它和 300/30 的结果无区别），
三组原有配置（300/30、600/60、1000/100）数值与修复前完全一致，说明修复没有副作用。

---

## 五、阶段 4：工程化包装

### 5.1 FastAPI 接口（端口 8020，svc.ps1 短名 `kbapi`）

| 端点 | 作用 | 是否需要 Key |
|---|---|---|
| `GET /health` | 状态 + Key 是否配置 + 语料规模 + 当前检索配置 | 否（缺 Key 也返回 200，状态为 `degraded`） |
| `GET /config` | 当前检索/分块/模型/路径配置，用于确认实验条件 | 否 |
| `POST /chat` | 问答：`{question, session_id}` → 答案 + 命中片段 + 分段耗时 | 是 |
| `GET /docs` | Swagger 自文档 | 否 |

错误语义（**不返回 500 堆栈，全部是可操作的明确错误**）：

| 状态码 | error | 触发条件 | 提示内容 |
|---|---|---|---|
| 503 | `api_key_missing` | 未配置 `DASHSCOPE_API_KEY` | 把 Key 填进 `.env` 后 `.\svc.ps1 restart kbapi` |
| 503 | `knowledge_base_empty` | 向量库片段数为 0 | 先跑 `scripts\ingest.py` 入库 |
| 502 | `upstream_error` | 模型服务调用失败 | 检查 Key 有效性/网络，或看 `_logs\kbapi.out` |
| 422 | — | 参数校验失败（如空问题） | Pydantic 标准结构 |

设计要点：

1. **`RagService` 懒加载**：服务能先起来，`/health` 随时可用，不会因为缺 Key 就启动失败；
2. **门禁顺序**：先查 Key（`/chat` 的 503 判断在 `import rag` 之前），
   这一点很关键——`DashScopeEmbeddings` 在**构造时**就校验 Key，若顺序反了会变成 500 堆栈；
3. 输出不用 emoji（Windows 控制台 GBK）。

### 5.2 ★ 致命缺陷：`RagService` 完全无法构造（上游遗留，本次发现并修复）

**现象**：`from rag import RagService` 直接抛

```
ImportError: cannot import name 'RunnableWithMessageHistory' from 'langchain_core.runnables'
```

**根因**：`langchain-core 0.1.53` 的 `runnables/__init__.py` **并不导出** `RunnableWithMessageHistory`
（实测 `dir(langchain_core.runnables)` 里没有它），正确路径是子模块
`langchain_core.runnables.history`。该行是**上游原项目的代码**（`git show HEAD~5:rag.py` 可见），
不是本次改造引入的。

**为什么说这不是偶发**：项目自己声明的依赖范围是 `langchain-core>=0.1.52,<0.2.0`，
在该范围内解析到的是 **0.1.53** —— 也就是说这个缺陷
**在项目自己声明的依赖范围内必然发生**，属于版本适配缺陷，不是环境偶然或本机特例。
（复现：`from langchain_core.runnables import RunnableWithMessageHistory` → MISS；
`from langchain_core.runnables.history import RunnableWithMessageHistory` → OK。）

**影响面（这才是真正严重的地方）**：

- `RagService` 是问答链路的入口，构造失败 = **整个系统无法回答任何问题**；
- 它是**潜伏缺陷**：Streamlit 首页能打开、`/health` 返回 200、评测在 mock 模式下全绿，
  因为 mock 模式走的是离线检索替身、**从不 import `rag`**；
  而 `/chat` 的 503 门禁又在 `import rag` 之前返回。
  → 直到阶段 4 用假 Key 触发 `RagService()` 构造，才把它逼出来。

**修复**：改用正确模块路径，并加注释说明为什么不能改回去。

**证据（假 Key 下构造，构造阶段不发网络请求）**：

```
[检索] 策略=vector，最终 k=1，重排=关
  [OK] vector 检索器 = HybridRetriever | chain = RunnableWithMessageHistory
[检索] 策略=bm25，最终 k=1，重排=关
  [OK] bm25 检索器 = HybridRetriever
[检索] 策略=hybrid，最终 k=1，粗排候选=10，重排=开（gte-rerank-v2）
  [OK] hybrid 检索器 = HybridRetriever | 重排器 = DashScopeReranker | 模型 = gte-rerank-v2
[OK] config_data / file_history_store / vector_stores / knowledge_base / retrievers / rag / server 全部导入成功
```

**教训**：`ImportError` 这类"整条链路不可用"的缺陷，必须靠**导入全量模块**的冒烟测试兜住，
只跑评测（走替身路径）是发现不了的。本项目的导入冒烟命令：

```powershell
.\.venv\Scripts\python.exe -c "import config_data, file_history_store, vector_stores, knowledge_base, retrievers, rag, server; print('全部导入成功')"
```

### 5.3 顺带修复：`/health` 路径漏掉遥测屏蔽

`server.py` 最初自己 `new` 了一个 `chromadb.config.Settings`，绕过了
`vector_stores` 里的遥测 logger 屏蔽，于是 `/health` 每被调用一次就往 stderr 刷一行
`Failed to send telemetry event ...`。现改为复用 `vector_stores.CHROMA_SETTINGS`
（导入即触发屏蔽）。实测 `_logs\kbapi.err` 已干净：

```
INFO:     Started server process [24660]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8020 (Press CTRL+C to quit)
```

### 5.4 指标口径说明（`report.py`）

新增「指标口径说明」小节，**逐条**写明每个指标在两种口径下的真实算法
（离线启发式 = 字符二元组重叠率，不调用任何模型；RAGAS = 裁判 LLM 判定）。
原因：同名指标在 mock 与 real 下算法完全不同，只看表头必然误读。
若一张表里混了两种口径，会额外插入「数值不可互相比较」的警告。

### 5.5 阶段 4 剩余项

- [x] a) FastAPI 包装（端口 8020，`/health` + `/chat`，缺 Key 返回明确错误）
- [x] b) 注册进 `svc.ps1`（短名 `kbapi`，BOM/CRLF/语法均校验通过）
- [x] d) CHANGES.md 划定「有效产出 / 暂无效结论」两栏（见 六、结论有效性划定）
- [ ] c) README 骨架（架构图 + 检索策略说明 + 模块结构 + 缺陷清单 + 启动方式）
- [ ] 接入真实嵌入：新增 `--embedding-backend dashscope|local`（local 指向本地
      `http://127.0.0.1:11434/v1/embeddings`，bge-small-zh-v1.5 / 512 维），
      并按「嵌入后端 + 维度」隔离 collection（等通知开始）
- [ ] 建 IR 指标：Hit Rate@k / MRR / Recall@k / nDCG@k（确定性、不需 LLM judge；等真实嵌入到位）
- [ ] 重排的真实收益数字（等嵌入与 IR 到位后再定，需 rerank 模型）
- [ ] 真实 `report_optimized.json` 与真实对比表
- [ ] 本地 LLM 替换云端（零成本对比，等通知）

### 5.6 本地嵌入服务接入规格（已实机核对，供第 3 步实施用）

> 这一节固化接入前的实测结论。**尤其注意 preflight 的路径**——
> 我最初写成 `GET /v1/health`，实测该路径不存在（404），若带进代码会造成最难查的故障：
> 探测永远 404 → 判定"本地服务不可用" → **本地嵌入一次都不会被启用，而且不报错**，
> 只是静默地永远走 dashscope 分支。

**端点实测（`curl` 实测于本机，2026-10-07）**

| 方法 | 路径 | 实测 | 说明 |
|---|---|---|---|
| GET | `/health` | **200** | 顶层健康检查，**preflight 用这个** |
| GET | `/v1/models` | 200 | 返回 `qwen2.5-1.5b-instruct`、`bge-small-zh-v1.5` |
| POST | `/v1/embeddings` | 422 | 缺 `input` 时返回 422（正常） |
| POST | `/v1/chat/completions` | — | 对话（本轮不用） |
| GET | `/v1/health` | **404** | ❌ 不存在，勿探测 |

`GET /health` 实测返回：

```json
{
  "status": "ok", "model": "qwen2.5-1.5b-instruct", "device": "cpu",
  "chat":      {"model": "qwen2.5-1.5b-instruct", "loaded": false},
  "embedding": {"model": "bge-small-zh-v1.5", "loaded": false, "available": true, "dim": null}
}
```

**四个必须避开的坑**

0. **探活与调用的根路径不同**（实施时实测踩到）：本地服务暴露两个根
   —— 探活用 `GET {base}/health`（顶层，**不带** `/v1`），
   调用用 `POST {base}/v1/embeddings`（**带** `/v1`）。
   OpenAI 客户端会在 `base_url` 后拼 `/embeddings`，所以给它的必须是带 `/v1` 的地址；
   写成裸 base 会得到 `404 {'detail': 'Not Found'}`（错误信息很含糊，容易误判成"服务没起"）。
   代码见 `embeddings.api_base_url()`。
1. **`embedding.dim` 在模型未加载时是 `null`**（上面实测就是 `null`），要等第一次嵌入调用之后才变成 512。
   → **不要**用它做 collection 维度校验，用下面的常量表（确定性、无副作用），
   再在第一次嵌入调用之后用真实向量长度复核一次，不一致就抛错。
2. **`loaded` 冷启动时是 `false`**（服务刻意做成懒加载——一次探活不该吃掉 200MB/3.1GB 内存）。
   → 判断可用性**只能**用 `available`，**不要**把 `loaded=false` 当成不可用。
3. **维度不一致绝不可混 collection**：DashScope `text-embedding-v4` 与本地 `bge-small-zh-v1.5` 维度不同，
   混用会让相似度结果完全是垃圾。要按「嵌入后端 + 维度」隔离 collection（如 `rag_local`），
   并把这两个字段写进 collection metadata，启动时不一致就报错退出。

**维度常量表**（校验用；DashScope 一列需在真实调用后确认）

```python
EMBEDDING_DIMS = {
    "dashscope": {"text-embedding-v4": 1024},   # 待真实调用复核
    "local":     {"bge-small-zh-v1.5": 512},    # 服务 /health 声明，实测待复核
}
```

**正确写法**

```python
LOCAL_BASE = "http://127.0.0.1:11434"

def preflight_local():
    try:
        import httpx
        r = httpx.get(f"{LOCAL_BASE}/health", timeout=3)   # ★ 顶层 /health，不是 /v1/health
        r.raise_for_status()
        info = r.json()
    except Exception as e:
        return False, (f"本地嵌入服务未启动（{e}）。\n"
                       f"启动：cd D:\\Projects ; .\\svc.ps1 start llm")
    emb = info.get("embedding", {})
    if not emb.get("available"):
        return False, "本地嵌入模型目录缺失（embedding.available=false）"
    return True, emb.get("model")
```

**复用现有机制**：报告 meta 需同时记录 `k / collection / 嵌入后端 / 维度`，
并把 5.1 里那条「k 不一致就拦截」扩展为「k、嵌入后端、维度任一不一致就拦截」。

### 5.7 接入真实嵌入（local 后端）+ 两个新发现的缺陷

#### 实现

| 内容 | 说明 |
|---|---|
| `embeddings.py`（新） | 嵌入后端 profile：解析 → local 探活 → **实测真实维度** → collection 指纹校验 |
| `config_data.py` | 新增 `embedding_backend`（默认 `dashscope`，现有行为不变）、`local_embedding_*`、`collection_override` |
| CLI 开关 | `scripts/ingest.py` / `evaluation/run_ragas.py` / `evaluation/experiments.py` 均新增 `--embedding-backend dashscope\|local` |
| 集合隔离 | 默认名按后端派生：`dashscope → rag`、`local → rag_local`；显式 `--collection` 可覆盖 |
| 指纹护栏 | 把「后端 + 模型 + 实测维度」写进 collection metadata，不一致**直接报错退出**（含可操作的换集合命令） |
| 报告 meta | `retrieval` 记录实际 `collection` / `embedding_backend` / `embedding_model` / `embedding_dim`；`report.py` 的拦截条件从「k 不一致」扩展为「k、后端、维度任一不一致」 |

#### ★ 新缺陷 1（上游遗留，**静默型**）：`metadatas` 拼成 `metadata`，元数据全部丢失

**证据**：入库 6 个片段后，逐个读回 metadata —— **6/6 全是 `None`**。

**根因**：LangChain 的签名是 `add_texts(texts, metadatas=None, ...)`（**复数**），
原项目写成 `metadata=`（单数），落进 `**kwargs` 被**静默忽略**。

**影响**：不知道片段来自哪个文档；`format_document` 拼给大模型的是空元数据；
**文档级归因（IR 指标、分块实验的按文档统计）完全做不了**。

**为什么不是偶发**：`**kwargs` 吞掉拼错的参数名是 LangChain 的既有行为，
任何环境、任何版本都必然发生，且**不报错、不告警**。

**修复**：改用 `metadatas=`，并顺带补上 `chunk_index`（便于定位命中的是文件里第几段）。
修复后实测：`{'chunk_index': 0, 'create_time': '2026-10-07 22:18:33', 'operator': '客户', 'source': '尺码推荐.txt'}`

#### ★ 新缺陷 2（**我在阶段 3 引入的回归**，同样是静默型）：Chroma 退化成进程内内存库

**现象**：入库进程自己 `count()` 得到 6 个片段，但**换一个进程** `get_collection('rag_local')`
直接 `ValueError: Collection rag_local does not exist.` —— 数据根本没落盘。

**根因**：`langchain_chroma.Chroma` 只有在**不传** `client_settings` 时才会自己构造
`Settings(is_persistent=True, persist_directory=...)`；一旦传入 `client_settings` 就照用不误、
**不会补 `is_persistent`**。而 chromadb 的 `Settings` 默认 `is_persistent=False`。
阶段 3 我为了关掉遥测报错传入了 `client_settings`，把这个默认值弄丢了 → 退化成内存库。

**症状为什么极隐蔽**：进程内自测（入库后立刻 count、同一进程检索）**100% 通过**，
只有「入库一个进程、检索另一个进程」的跨进程验证才会暴露 ——
而 `/health` 之前一直显示 `documents: 0`，我原以为是"语料还没入库"，其实是磁盘库与写入库不是同一个。

**修复**：`CHROMA_SETTINGS` 显式加上 `is_persistent=True, persist_directory=...`。

**证据（修复前 vs 修复后，同一命令、新进程读取）**：

```
修复前：磁盘上的集合 = ['rag']            ← rag_local 不在，6 个片段只存在于内存
        ValueError: Collection rag_local does not exist.
修复后：磁盘上的集合 = ['rag', 'rag_local']
        rag_local 片段数 = 6
        嵌入指纹 = {'embedding_backend': 'local', 'embedding_dim': 512, 'embedding_model': 'bge-small-zh-v1.5'}
```

#### 首次真实向量检索验证（local / 512 维 / 6 片段）

（`--embedding-backend local --rebuild` 入库，跨进程读取，`similarity_search_with_score` 打分）

| 问题 | 期望文档 | 向量 top3（`HIT` = 命中期望文档） |
|---|---|---|
| 怎样穿显得更瘦？ | 颜色推荐.txt | **HIT 0.9275 颜色推荐** / HIT 0.9619 颜色推荐 / 1.0377 洗涤养护 |
| 身高168cm、体重100斤，建议穿什么尺码？ | 尺码推荐.txt | **HIT 0.5018 尺码推荐** / 1.0797 颜色推荐 / 1.0961 颜色推荐 |
| 真丝连衣裙应该怎么洗？ | 洗涤养护.txt | **HIT 0.8194 洗涤养护** / HIT 0.8325 洗涤养护 / HIT 0.9087 洗涤养护 |

同一组问题用 BM25 的 top1 对比：

| 问题 | 向量 top1 | BM25 top1 |
|---|---|---|
| 怎样穿显得更瘦？ | 颜色推荐.txt（命中） | **无（0 命中）** |
| 身高168cm、体重100斤，建议穿什么尺码？ | 尺码推荐.txt | 尺码推荐.txt |
| 真丝连衣裙应该怎么洗？ | 洗涤养护.txt | 洗涤养护.txt |

**这一步真正结束了一个悬案**：4.5 第 2 条曾猜测「语义改写是字面检索的死角，只有真向量能救」，
当时只是推测（离线替身里两路都是字面检索）；现在用**真嵌入 + 真语料**复现了它 ——
`怎样穿显得更瘦？` 向量能命中、BM25 直接 0 命中。

#### 现在哪些结论有效了（对应第六节的更新）

- ✅ **新有效**：「真向量检索能解决字面检索的语义死角」—— 单样本级证据，可复现（上表）。
- ✅ **新有效**：元数据归因可用（修复缺陷 1 之后），这是 IR 指标的前置条件。
- ⚠️ **仍不够**：上表只有 **3 个问题**，n=3 不构成统计结论；
  必须把 29 条评测集全跑一遍 + 上确定性 IR 指标，才能下"hybrid 优于纯 BM25"这类判断（第 4 步）。
- ⚠️ **仍无效**：RAGAS 4 指标的真实数值（需要 LLM judge）；分块参数对比。

---

## 六、结论有效性划定（写 README / 简历 / 汇报前必读）

> 这一节把「哪些结论站得住、哪些站不住」一次性划清楚，避免下游文档误把替身数据当成果。
> 判定标准只有一条：**结论里的「检索」和「打分」是否都是真的。**
> 只要有一方是替身，该结论就不具备可迁移性。

### 6.1 有效产出（可以作为结论）

| # | 产出 | 为什么有效 | 证据落在哪 |
|---|---|---|---|
| 1 | 检索层代码：向量 / BM25 / RRF 混合 / 重排，统一 `invoke()` 契约 | 代码路径已验证（不涉及效果判断） | 8 项探针全通过，含「`vector` + `k=1` 只返回 1 条」的回归 |
| 2 | `chunk_size` 对短文档失效缺陷的修复 | **客观事实**：片段数 1 → 2 | 4.8：切分对照（1 → 2，[133,135]）+ 真实入库链路端到端（`FakeEmbeddings`，去重未破坏） |
| 3 | 片段数统计 | **客观事实**：纯本地切分计数 | 4.8 全量扫描：150/15→32、300/30→15、600/60→9、1000/100→6 |
| 4 | 机制性护栏之一：k 不一致拦截 | 机制存在且实测触发 | 4.3 结尾：k=1 vs k=4 对比时终端与 Markdown 均输出口径警告 |
| 5 | 机制性护栏之二：遥测治理 | 机制存在且实测生效 | 5.3：`_logs\kbapi.err` 已无 `Failed to send telemetry event` |
| 6 | ★ 致命 ImportError 的定位与修复 | **客观事实**：修复前 `import rag` 直接失败 | 5.2：修复前 ImportError 原文 + 修复后三策略构造成功 |
| 7 | 修复 2 个**静默型**缺陷：`metadatas` 拼写错误、Chroma 持久化回归 | **客观事实**：前者修复前 6/6 片段 metadata 为 `None`；后者修复前跨进程找不到集合 | 5.7：修复前后对照 + 跨进程读取证据 |
| 8 | 嵌入后端可切换且有护栏（`dashscope` / `local`） | 机制存在且实测：探活失败、后端不一致、维度不一致三种情况均按预期拦截与提示 | 5.6 / 5.7：护栏实测输出 |
| 9 | 真向量检索能命中字面检索完全漏掉的样本（**能力验证**，非效果结论） | 单样本级、可复现 | 5.7：`怎样穿显得更瘦？` 向量 top1 = 颜色推荐.txt，BM25 = 0 命中 |
| 10 | IR 指标模块（Hit Rate@k / Precision@k / Recall@k / MRR / nDCG@k，确定性、**不需要 LLM 裁判**） | 数学正确性经**手算校验**，8/8 一致 | 4.3-R 上方：`hit_rate@1=0.5 / mrr=0.6667 / ndcg@5=0.75` 等全部与手算相符 |
| 11 | 用真实嵌入 + IR 指标重做 4.3 / 4.4（替身表的替代品） | 真实嵌入（512 维）+ 确定性指标 + 多设置复现 | 4.3-R / 4.4-R |
| 12 | BM25 在文档级稳定落后于 vector / hybrid | 三个不同设置下方向一致（每次差 1 题）；**幅度在噪声内，不宣称显著** | 4.3-R：0.8966 vs 1.0000（1000/100）、0.9655 vs 1.0000（300/30、150/15） |
| 13 | 分块越细，top-k 里的噪声越少 | 单调且可复现，且刻意选用了不受语料粒度污染的 precision | 4.4-R：`chunk_precision@3` 0.9080（150/15）→ 0.6092（1000/100） |

这 13 条的共同点：**结论都配了可复现的证据，并且把局限写在结论旁边。**
（第 9 条是"能力验证"；第 12 条幅度在噪声内；第 13 条只覆盖"噪声"这一个维度 —— 都不越界宣称。）

### 6.2 暂无效结论（必须等真实嵌入复测，**绝对不要进 README 结论区**）

| # | 结论 | 位置 | 为什么无效 |
|---|---|---|---|
| 1 | 三路检索策略对比（`vector` / `bm25` / `hybrid` 的 recall / precision / relevancy） | 4.3 表 | mock 下「向量腿」是字面检索替身 → `hybrid` = 字面 + 字面，**两路同源**，比较不出混合检索的真实增益 |
| 2 | 四条分块参数对比表（150/15 ~ 1000/100 的**全部指标**） | 4.4 表 | mock 下「检索」与「指标」**都是替身**，两个替身相乘，结论没有可迁移性 |

必须一起讲清楚的补充说明，否则容易被断章取义：

1. 4.3 那条曾被我写成「hybrid 召回 +7.3%」，**该结论已作废**：那个增益来自 `k` 从 1 提到 4，
   与混合检索无关。k 对齐后三路召回极差仅 3.2%，且 `hybrid` 的 precision 反而最低（-6.4%）。
2. 这两张表**保留在文档里是有意的**，但它们的作用只是「证明评测流程可跑通」，
   不是效果证据——4.3 / 4.4 的标题下方已各自固定插入「不可作为结论，需真实嵌入复测」的横幅。
3. 一个可复用的判据：`hybrid` 与 `bm25` 在 mock 下数值几乎相同（0.7481 vs 0.7159，且 precision 更差），
   **这本身就说明混合里的"向量腿"没提供新信息** —— 正是"替身"的直接证据。
4. 复测口径：真实嵌入（DashScope，或本地 `bge-small-zh-v1.5` / 512 维）
   + 至少一种不依赖 LLM judge 的确定性指标（Hit Rate@k / MRR / Recall@k / nDCG@k），
   产出后才可以写进 README 的结论区。
5. 复测时还需按「嵌入后端 + 维度」隔离 collection（不同维度绝不能混用），
   并把这两个字段写进报告 meta，沿用已有的「口径不一致就拦截」机制。**这一条已于 5.7 实现。**
6. ~~真嵌入已接入，但上面两张表仍无效~~ → **已用真实嵌入 + 确定性 IR 指标重做，见 4.3-R / 4.4-R。**

   ⚠️ 但请连带引用它们的新局限（否则等于换一种方式夸大）：
   - **文档级指标在 3 份文档上近乎饱和**（`doc_hit@1` 普遍 1.0），区分度有限；
     `doc_precision@k` 因文档数只有 3 而退化，不可解读。
   - **片段级是弱标注口径，且结构性偏袒嵌入型检索器**（正例由嵌入相似度定义），
     所以 `chunk_*` 上 vector/hybrid 领先 BM25 是预期内的口径偏袒，**不是独立证据**。
   - **n=29，差 1 题 = 3.4%，不构成统计显著**。可复现的是"方向一致"，不是"显著更优"。
   - 4.3 / 4.4 那两张**替身表本身仍然无效**（检索与打分都是替身），
     它们只作为"评测流程可跑通"的历史记录保留。

### 6.3 用词红线

- ❌ 不要把 4.3 / 4.4 的任何数字写成「RAGAS 指标」——
  mock 模式下 4 个指标全是**字符二元组重叠率**，与 RAGAS 官方算法无关（逐条口径见报告的「指标口径说明」小节）。
- ❌ 不要把 4.3 / 4.4 的数字写进 README 的「结论 / 效果」区，也不要写进简历。
- ✅ 可以写的是：改了哪些代码、修了哪些缺陷、建了哪些机制、以及客观的片段数与耗时统计。

---

## 七、禁止事项（改造红线，勿破）

1. 不重建 `.venv`，不换成 Python 3.12（`chroma-hnswlib==0.7.3` 在 3.12 上只能现场编译 C++，会失败）。
2. 不改 `requirements.txt` 的版本锁定。
3. `requirements.txt` 含中文注释，直接 pip 装会 `UnicodeDecodeError: 'gbk'`，需先 `$env:PYTHONUTF8='1'`。
4. `DASHSCOPE_API_KEY` 只放 `.env`，禁止写进代码或提交。
5. 不下载任何 GitHub Release 文件（通道被阻断）。
