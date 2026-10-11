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

条件：分块 **1000/100**（语料 6 片段）、`top_k=4`、候选 10、local 512 维嵌入。

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

### 4.3-R 附录：为什么"弱标注偏袒"在比检索器时是致命的，在比分块时却无害

同一条"片段级弱标注"口径，用在两处实验里的**性质完全不同**，这决定了结论能有多强：

| | 4.3-R 比**检索器** | 4.4-R 比**分块配置** |
|---|---|---|
| 弱标注标准 | 嵌入相似度 ≥ 阈值 | 嵌入相似度 ≥ 阈值（**同一标准**） |
| 被比较对象是否也是嵌入驱动 | **是**（vector / hybrid 本质靠嵌入排序） | 否（4 个配置用的是**同一个** hybrid） |
| 偏差性质 | **结构性偏袒**：被测对象与标注标准同源 | **常量偏差**：四档受到同样的偏袒 |
| 对组间比较的影响 | **破坏**（把"同源"错当成"更准"） | **不破坏**（偏差在组间约掉） |
| 可用于结论吗 | ❌ 只能当辅助线索 | ✅ 单调趋势可信 |

★ **所以结论强度排序是：4.4-R 的分块规律 > 4.3-R 的检索器胜负。**
再叠加一层：4.4-R 是**四档一致单调**（0.6092 → 0.7011 → 0.7701 → 0.9080），
而 4.3-R 是"差 1~3 题"。**单调性不是靠样本量堆出来的**，在 n=29 下反而比"差 1 题"有说服力得多。

因此可以写进结论区的是这一条（可操作的工程结论）：

> **粗分块会把不相关的内容混进同一个片段，导致 top-k 里的噪声比例系统性上升**
> —— 在同一检索器（hybrid）与同一 k（=4）下，四档分块的 `chunk_precision@3`
> 从 0.9080（150/15）单调降到 0.6092（1000/100）。

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

### 4.9 扩语料复测：**分块规律没有被复现**，并发现第二个口径陷阱

#### 4.9.1 为什么扩语料 + 语料怎么来的

3 份文档 + n=29 结构上不可能得到统计显著，继续追"hybrid 是否更优"只会得到一个站不住的东西。
所以目标改为：**验证 4.4-R 的分块规律在更大语料上是否仍成立**。

语料（`corpus/`，12 篇、合计 **225,812 字**，抓取脚本 `scripts/build_corpus.py`）：

| 项 | 内容 |
|---|---|
| 来源 | `docs.python.org/zh-cn/3`（Python 官方文档中文翻译），逐页取正文 div |
| 许可 | PSF License；署名与用途说明见 `corpus/SOURCES.md` |
| 选材原则 | ① **题材中性**（纯技术文档，刻意避开本机含个人信息的规划类文档）；② **主题互相独立**（12 个主题，使文档级指标有区分度） |
| 噪声治理 | 抽取时剔除 `toctree-wrapper`（内嵌目录）与侧栏/导航 —— 这些文字在 12 篇里**逐字相同**，留着会让文档级评测失效 |

#### 4.9.2 评测集怎么造的（ground truth 的构造方式决定结论可信度）

脚本 `evaluation/build_eval_set.py` → `evaluation/eval_set_v2.jsonl`。

| 项 | 内容 |
|---|---|
| 谁写的 | 由我（AI 助手）撰写，**不是人工标注**；本文件把这一点显式写出来 |
| 规模 | 36 题，12 篇文档 × 3 题；每题都给了 `answer_keywords` 用于程序化校验 |
| 出题原则 | 每题必须能用来源文档里的内容回答；`ground_truth` 尽量沿用文档用词（保证可判定） |
| **语义改写比例** | **实测 38.9%（14/36）** —— 定义：问题的内容词与来源文档的重叠率 < 0.6 |
| 度量方法 | jieba 分词 → 去标点/纯数字/单字/停用词 → 计算"问题内容词中出现在来源文档里的比例"；**英文标识符保留**（Python 文档里它们是最强检索信号，只算中文会高估改写比例） |
| 作者意图 vs 实测 | 我按"每题 1 道改写题"的意图出题，但**实测有 8 题与意图不符**，一律以实测为准（`question_type` 为实测值，`question_type_authored` 仅留痕） |
| 度量口径的保守性 | 重叠率是拿问题比**整篇文档**，文档越长越容易命中 → 真实改写比例**不低于** 38.9% |

**关键词校验拦住了 2 个真问题**（这就是它存在的意义）：

1. `intro-03` 我写"三引号"，但文档用的是**"三重引号"** → 已改；
2. `flow-02` 原拟问 `*args/**kwargs`，但实测该文档 `kwargs` 出现 **0 次**
   （4.9.2 讲的是**调用**时的关键字参数）→ 已换成该文档确实覆盖的「可变默认值陷阱」。

#### 4.9.3 复测结果：分块规律**没有复现**

命令：

```powershell
.\.venv\Scripts\python.exe evaluation\run_ir.py --name ir2_chunk --corpus corpus ^
    --configs 150/15,300/30,600/60,1000/100 --retriever hybrid --top-k 4 --candidate-k 32 ^
    --embedding-backend local --eval-set evaluation\eval_set_v2.jsonl
```

| 指标 | 150/15 | 300/30 | 600/60 | 1000/100 | 与 4.4-R 一致？ |
|---|---|---|---|---|---|
| 语料片段数 | 1941 | 872 | 422 | 252 | — |
| doc_hit_rate@1 | 0.6667 | **0.7222** | 0.6944 | 0.6389 | ❌ 非单调 |
| doc_mrr | 0.7593 | **0.8194** | 0.7685 | 0.7593 | ❌ 非单调 |
| chunk_precision@3 | 0.6574 | 0.6389 | 0.6204 | **0.6759** | ❌ **反转**（最粗的反而最高） |
| chunk_precision@5 | 0.5111 | 0.5167 | 0.5111 | 0.5222 | ❌ 基本持平 |

**4.4-R 那条"分块越细噪声越少（四档单调）"的结论，在 12 篇文档上没有复现，反而出现反转。**

#### 4.9.4 ★ 发现第二个口径陷阱：弱标注的**基准率会随分块变粗而上升**

| 分块配置 | 语料片段数 | 平均正例/query | 正例占语料比例（基准率） |
|---|---|---|---|
| 150/15 | 1941 | 174.86 | **9.0%** |
| 300/30 | 872 | 87.89 | **10.1%** |
| 600/60 | 422 | 52.92 | **12.5%** |
| 1000/100 | 252 | 38.06 | **15.1%** |

机制很直白：分块越粗，单个片段平均包含越多"与标准答案相似"的内容，
于是被弱标注判为相关的片段**占比**越高。而 `precision@k` 的下界就是这个基准率 ——
**基准率自己涨了 1.7 倍，precision 的"提升"里有多少是检索真的变好，无法区分。**

#### 4.9.5 ★ 这同时推翻了我上一轮给出的"结论强度排序"

我在 4.3-R 附录里写过：弱标注偏差在"比分块"时是**常量**、在"比检索器"时是**结构性偏袒**，
因此分块结论更强。**这个论证的前提被上面的数据否定了**：

| | 4.3-R 比检索器（**同一**分块配置） | 4.4-R 比分块配置 |
|---|---|---|
| 弱标注基准率 | **三路完全相同**（同一配置、同一套标注） | **随分块变粗从 9.0% 漂到 15.1%** |
| 偏差性质 | 只剩"标注生成器与向量检索同源"这一条**循环性** | 除循环性外，**还叠加了基准率漂移** |
| 结论强度 | 可作**佐证**（基准率不漂，且三路差距大） | ❌ **被基准率漂移污染** |

**修正后的排序：在同一分块配置下比检索器 > 跨分块配置比 precision。**
上一轮把两者判反了 —— 判据本身没错（"偏差是否组间相同"），错在我没有**实测**基准率是否真的相同。

#### 4.9.6 扩语料后有效的结论（12 篇文档、36 题、固定 300/30、k=4、候选 10）

| 指标 | vector | bm25 | hybrid |
|---|---|---|---|
| doc_hit_rate@1 | **0.6389** | 0.5000 | 0.6111 |
| doc_mrr | **0.7269** | 0.6134 | 0.7153 |
| doc_ndcg@5 | **0.7543** | 0.6551 | 0.7524 |
| chunk_precision@3（基准率三路相同） | **0.6574** | 0.4167 | 0.6111 |
| chunk_recall@3 | 0.0515 | 0.0491 | 0.0427 |

- ✅ **语料变大后文档级指标终于有区分度**（0.5 ~ 0.72，不再是清一色 1.0）：
  BM25 从 23/36 掉到 18/36（0.6389 vs 0.5000），**这是目前最像结论的一条**。
- ⚠️ 但 **n=36、差 5 题**，仍不足以宣称统计显著；且 vector 略优于 hybrid（0.6389 vs 0.6111）
  与"混合检索必然更好"的直觉相反 —— 如实记录。
- ⚠️ **`candidate_k` 也是变量**：同一配置（hybrid + 300/30）在 `candidate_k=32` 时 doc_hit@1=0.7222，
  在 `candidate_k=10` 时=0.6111。**跨报告比较前必须同时对齐 k 与 candidate_k。**
- ⚠️ 弱标注基准率过高（9%~15%）说明阈值 0.6 对 bge-small 偏松，
  `chunk_*` 这一层目前只能当粗线索；要变严格需要更高阈值或改用"lift = precision@k / 基准率"这类归一化指标。

### 4.10 第三个口径 + lift 归一化：两个独立口径指向同一方向 ★

> **⚠️ 本小节的数字已被 §4.11.3 取代。** 4.10.2 / 4.10.3 两张表都是**单次运行**，
> 而本轮实测出向量腿跨进程不确定（噪声地板约 3 题 / 36，见 4.10.5、4.11）。
> 这里保留它们**作为"噪声地板"这一发现的证据**，但**引用数字请用 4.11.3 的 3 次均值表**。
> 另：4.10.3 里"`kw_precision@1` 领先"这一条已在 4.11.3 降级（余量不足地板自身不确定度）。

针对 4.9.4 暴露的"基准率漂移导致跨分块不可比"，这一轮补了两件事：

| 补的东西 | 作用 | 不解决什么 |
|---|---|---|
| `lift@k = precision@k / 基准率` | **跨分组可比性**：基准率不同的两组可以横向比 | **解决不了标注的循环性** |
| **对抗性口径**（`kw_*`）：片段含该问题的 `answer_keywords` 即算相关（**纯词面，不含嵌入**） | **切断循环性**：标注不再由嵌入模型生成 | 不解决基准率漂移（它自己也会漂，所以也要配 lift） |

> 阈值保持 0.6 **不收紧**：收紧的收益是基准率下降、lift 区分度上升；
> 代价是留下的正例恰好是嵌入模型最"自信"的那批 → 与标注生成器同源更强 → **循环性加重**。
> 而"比检索器"时三路基准率完全相同，收紧只付代价、换不到可比性收益。理由已写进 `run_ir.py` 注释。

#### 4.10.1 三口径定义

| 前缀 | 标注方式 | 是否含嵌入 | 循环性 | 已知偏袒 |
|---|---|---|---|---|
| `doc_` | 评测集 `relevant_sources`（人工指定的应命中文档） | 否 | 无 | — |
| `kw_` | 片段文本含该问题的 `answer_keywords`（词面匹配） | **否** | **无** | 可能偏袒关键词检索 |
| `chunk_` | 与标准答案嵌入相似度 ≥ 0.6 | 是 | **有**（与向量检索同源） | 向量 / 混合 |

#### 4.10.2 分块对比：raw precision 不可比，lift 归一化后两口径**一致单调**

语料 12 篇（22.6 万字）、36 题、`hybrid`、`k=4`、候选 32：

| 配置 | 片段数 | chunk 基准率 | chunkP@3 | **chunkLift@3** | kw 基准率 | kwP@3 | **kwLift@3** |
|---|---|---|---|---|---|---|---|
| 150/15 | 1941 | 9.01% | 0.6944 | **7.708** | 6.92% | 0.5556 | **8.033** |
| 300/30 | 872 | 10.08% | 0.6389 | 6.339 | 10.17% | 0.5648 | 5.555 |
| 600/60 | 422 | 12.54% | 0.6111 | 4.873 | 14.42% | 0.6111 | 4.237 |
| 1000/100 | 252 | 15.10% | 0.6759 | **4.476** | 18.51% | 0.6667 | **3.602** |

**这张表同时解释了 4.9.3 那个"反转"**：

- **raw `precision@k` 的方向由基准率主导**：基准率从 9.01% 涨到 15.10%（chunk）/ 6.92% 涨到 18.51%（kw），
  而 `precision@k` 的**下界就是基准率** → 越粗的两档 raw precision 反而更高（甚至出现 1000/100 最高）。
  **这不是检索变好，是"随机也能蒙对"的概率变高了。** 4.9.3 的"反转"到此有了完整解释。
- **归一化后两个口径都严格单调**：`chunkLift@3` 7.708→6.339→4.873→4.476，
  `kwLift@3` 8.033→5.555→4.237→3.602。
  两个**标注机制完全不同**的口径（一个用嵌入、一个纯词面）给出**同向且严格单调**的结果 ——
  这比单一口径的单调可信得多。

**但要如实说明 lift 的性质**（避免又一种过度解读）：

- `lift` 是**相对随机基线**的倍数，不是"绝对效果"。分块越粗，随机命中率越高，
  检索相对于随机的领先幅度自然缩小 —— 单调下降里有相当一部分是这个机制造成的。
- 它**不推翻**"粗分块每个片段承载更多相关内容"这个事实（recall 型指标仍然偏向粗分块）。
- 所以正确表述是：**分块是权衡，但它不是"没有规律"** —— 在相对基线的口径下，
  越细的分块越能把噪声挡在 top-k 之外。

#### 4.10.3 检索器对比：对抗性口径下 vector 仍然领先 BM25 ★

12 篇文档、36 题、分块 300/30、`k=4`、候选 10（三路完全同条件）：

| 指标 | vector | bm25 | hybrid |
|---|---|---|---|
| `doc_hit_rate@1`（人工标注，无循环） | **0.6389** | 0.5000 | 0.6389 |
| `kw_precision@1`（**无嵌入**，可能偏袒关键词检索） | 0.5556 | 0.5000 | **0.5833** |
| `kw_precision@3` | **0.5463** | 0.5093 | 0.5278 |
| `kw_lift@3` | **5.373** | 5.009 | 5.191 |
| `chunk_precision@3`（含循环性，偏袒向量） | **0.6574** | 0.4167 | 0.6019 |
| `chunk_lift@3` | **6.522** | 4.134 | 5.972 |

**结论（这次可以硬一点了）**：在**刻意不含嵌入**的对抗性口径下，BM25 依然是最低的
（`kw_precision@1` 0.5000 vs 0.5556、`kw_precision@3` 0.5093 vs 0.5463、`kw_lift@3` 5.009 vs 5.373）。
也就是说 **"BM25 落后"不是循环性造出来的** —— 它在一个对向量不利的口径下依然成立。

必须一起说的限制：

- **差距仍然很小**：`kw_precision@1` 差 2 题 / 36，`kw_precision@3` 差约 1.3 题 / 36。
  方向一致（三个口径、两次实验都是 vector ≥ bm25），但**不足以宣称统计显著**。
- **hybrid 不是稳赢**：它在 `kw_precision@1` 上最好（0.5833），
  在 `kw_precision@3` / `kw_lift@3` / 两个 `chunk_*` 上都低于 vector。
  "混合必然更好"在这份语料上**得不到支持**，如实记录。
  ⚠️ 更要紧的是：**vector 与 hybrid 的差距（0~1 题）落在 4.10.5 实测的 2 题噪声地板之内**，
  所以准确说法不是"hybrid 略差"，而是**这两者无法排序**。
- 机制上仍然一致：BM25 的短板集中在语义改写类问题（问题与原文无共同关键词时字面匹配直接漏检）。

#### 4.10.4 机制化：口径不一致的拦截范围扩大

`report.py` 的口径校验从"只看 k"扩展为四项：**`k` / `candidate_k` / 嵌入后端+维度 / 重排开关**，
任一不一致就在终端与 Markdown 里警告（起因：实测同一配置 `candidate_k=32` 与 `=10`
的 `doc_hit@1` 分别是 0.7222 / 0.6111，candidate_k 是个看不见的变量）。

#### 4.10.5 ★ 评测噪声地板：向量腿跨进程不稳定（本轮最该被记住的一条）

**发现过程**：同一配置（hybrid + 300/30 + `k=4` + 候选 10）两次运行，`doc_hit@1` 是 0.6389 / 0.6111 —— 差 1 题。
先排除数据侧原因：

| 检查 | 结果 |
|---|---|
| 同进程内两次检索是否一致 | 一致（**0/36 题**不同）→ 代码本身是确定的 |
| 跨进程语料顺序（BM25 的语料来源） | 3 个进程哈希完全相同 → 稳定 |
| **跨进程 BM25 排序** | 哈希相同 → **确定** |
| **跨进程向量腿排序** | 哈希**每次都不同** → **不确定** |

```
进程1  向量腿=eace89c251af   BM25=d09fdb48abb2   hybrid=15955e9355cd
进程2  向量腿=b1f84f7333f2   BM25=d09fdb48abb2   hybrid=09d3cc7998c8
                             ^^^^^^^^^^^^^^^^^ 只有向量腿在变
```

**根因**：chromadb 0.4.24 的向量检索走 HNSW **近似**最近邻，索引图是在进程内重建的，
对"近似并列"的候选跨进程会返回不同顺序；BM25 是精确打分，因此不受影响。

**实测噪声地板**（同一配置跑 5 个独立进程，均 `--no-ingest`、同集合、同评测集）：

| 运行 | doc_hit@1 | doc_mrr |
|---|---|---|
| 第 1 次 | 0.6389 | 0.7106 |
| 第 2 次 | 0.6111 | 0.7060 |
| 第 3 次 | **0.6667** | **0.7361** |
| 第 4 次 | 0.6111 | 0.7083 |
| 第 5 次 | 0.6111 | 0.7060 |
| **极差** | **0.6111 ~ 0.6667 = 2 题 / 36** | 0.7060 ~ 0.7361 |

> 口径提醒：**不要把 `ir3_chunk` 那次算进来**——它用的是候选 32，与本组不是同一条件。
> （我第一版汇总时混进去了，得到"3 题"的错误极差，已改正为 2 题。）

**逐个复核已有结论（这是这条发现真正的用处）**

| 结论 | 差距 | 是否超过 2 题噪声地板 | 判定 |
|---|---|---|---|
| BM25 落后（`doc_hit@1`） | 5~6 题（0.5000 vs 0.6111~0.6667） | 是 | ✅ **成立** |
| BM25 落后（对抗口径 `kw_precision@1`） | 2 题（0.5000 vs 0.5556） | 刚好在边界 | ⚠️ 方向一致，**不能单独当证据** |
| vector vs hybrid 谁更好 | 0~1 题 | 否 | ❌ **无法排序**（在噪声地板之内） |
| 分块 `lift@3` 单调（7.708→4.476，降幅 3.2） | 单点抖动约 ±0.3~0.5 | 是 | ✅ **成立**（降幅远大于单点抖动） |

**所以**：这个项目的对比实验**必须报噪声地板**，否则 1 题级差异会被当成发现。
上面这张表也解释了 4.10.3 里"hybrid 不是稳赢"的表述为什么不准确 ——
不是"hybrid 略差"，而是**两者差距根本在噪声之下、无法排序**。

**下一步建议（未做）**：给 `run_ir.py` 加 `--repeat N`。
注意：**同进程内重复测不出这个噪声**（HNSW 图在同进程里稳定），必须起 N 个独立进程；
或者固定 hnswlib 的随机种子（要改 chromadb，风险大，暂不做）。

### 4.11 重复测量机制，与"确定性改造"的负结果 ★

#### 4.11.1 `--repeat N`：把"1 题级差异"变回可见（默认 3）

新增在 `run_ir.py`：

| 设计 | 做法 | 理由 |
|---|---|---|
| 必须独立进程 | `subprocess` 起 N 个子进程，不是 multiprocessing / 循环调用 | 噪声来自 HNSW 索引图**在进程内重建**，同进程重复测不出来（实测同进程 0/36 差异）。subprocess 也与手工测噪声的方式一致 |
| 每配置各自测地板 | 每个配置分别重复、分别记录，不做任何跨配置复用 | 实测 ck=32 时四组地板是 0.3 / 1.0 / 0.3 / 1.0 题，ck=10 时是 3.0 题 —— 确实是**每配置属性** |
| 报告可机读 | `meta.repeat_n` + `meta.noise_floor{value, items, headline_metrics, by_metric}`；指标取 N 次均值 | 极差同时给小数与题数（"2 题 / 36"才可操作） |
| ★ 报告层**自动拦截** | 对比表逐行判断，超不出该指标地板的一律标注，不需要人去记得 | 与"口径不一致拦截""白名单防漂移"同一条路子 |

标注分两级（这是第一版没做对的地方）：

- 一行里**最宽的一对**都 ≤ 地板 → `[本行最宽差异也在噪声内，无法排序]`
- 只有**首末两组** ≤ 地板（行内其它组仍可分）→ `[首末差异在噪声内]`
- 为什么按"最宽的一对"而不是只看 Δ(末-首)：三列之间有 **3 对**比较，只看首末会把
  BM25↔vector 的差距藏起来（BM25 在中间列，Δ 列根本看不到它）。
- 实测（**逐行数源表**，不是数过滤后的输出）：检索器对比表共 **39 行** ——
  **5 行**整行不可排序、**26 行**仅首末不可分、**8 行**没有标注（= 组间可分）。
  8 行按口径分：`kw_precision@3` / `kw_lift@3` 属**对抗口径（无嵌入）**；
  `chunk_precision@3` / `chunk_precision@5` / `chunk_lift@3` / `chunk_lift@5` /
  `chunk_recall@5` / `chunk_ndcg@5` 属**弱标注口径（嵌入同源）**。

> ★ **机制的能力边界（别让读者以为筛选是完备的）**
> 这个自动标注筛的是**噪声**（统计可比性）；它**筛不了口径质量** —— 循环性不是统计问题。
> 所以"这 8 行可分"≠"这 8 行都能当证据"：其中 6 行是嵌入同源口径，对 BM25 **结构性不利**。
> **口径分层必须由人做**，见 4.11.3。
>
> 报数纠错留痕：本节初版写的是"35 行 / 5 / 26 / 4"，那是从**过滤后的终端输出**里数出来的
> （只看到匹配到关键词的行）—— 与 4.10.5 把 `candidate_k=32` 的样本混进噪声地板是同一类错误：
> **报数必须回到源数据逐行数**。已更正为 39 / 5 / 26 / 8。

> 踩到的坑：第一版把"所有指标极差的最大值"当统一地板，冒烟时算出 **11.5 题 / 6** 这种荒唐数 ——
> `lift` 的量纲是 4~8、`hit_rate` 是 0~1，混在一起取最大值必然被 `lift` 主导。
> 已改为：逐指标用 `by_metric`；对外 headline 只取主指标
> （`doc_hit_rate@1` / `doc_mrr` / `kw_precision@3` / `chunk_precision@3`）里的最大值。

#### 4.11.2 确定性 tie-break 的负结果（有价值）

按建议试了"并列时按片段指纹排序"：`VectorRetriever` 显式按 `(距离, 片段指纹)` 排序
（原来直接信任 Chroma 返回顺序，拿不到距离就无从稳定）、RRF 按 `(-分数, 指纹)`、重排按 `(-分数, 下标)`。
改造后重测跨进程：

| 检查项 | 结果 |
|---|---|
| 查询向量本身跨进程 | **逐位一致**（36 条哈希全同；后端是本地 OpenAI 兼容服务，没有 torch 的线程抖动） |
| 语料顺序跨进程 | 一致（3 个进程哈希相同） |
| 改造后**顺序不同但集合相同**的题数 | **0** |
| 改造后**集合本身就不同**的题数 | **24 / 36** |

**结论：噪声不是"并列时顺序随机"，而是 HNSW 返回的候选集合本身在变。**
所以 tie-break 改造**消不掉**这个噪声 —— 它只把"顺序"这一半确定化了。

仍然保留这个改造，理由是它让诊断干净：改造后"仅顺序不同"= 0，
剩下的差异**全部**是集合级，于是"披露并测量"从备选方案变成唯一方案（已经没有别的东西可修了）。
这也是典型的"负结果但有用"。

（BM25 那条腿刻意没动：它是精确打分，实测跨进程排序哈希完全一致，本来就不需要 tie-break。已加注释。）

#### 4.11.3 三口径在 3 次均值下的结论（最终数字）

**分块对比**（12 篇、36 题、hybrid、k=4、候选 32、3 次独立进程均值 ± 该指标实测极差）：

| 指标 | 150/15 | 300/30 | 600/60 | 1000/100 |
|---|---|---|---|---|
| 语料片段数 | 1941 | 872 | 422 | 252 |
| 该配置噪声地板 | 0.3 题 | 1.0 题 | 0.3 题 | 1.0 题 |
| raw `chunk_precision@3` | 0.6698 (±0.009) | 0.6327 (±0.009) | 0.6111 (±0.000) | **0.6759 (±0.000)** |
| **`chunk_lift@3`** | **7.435 (±0.102)** | 6.278 (±0.092) | 4.873 (±0.000) | 4.476 (±0.000) |
| raw `kw_precision@3` | 0.5617 (±0.009) | 0.5679 (±0.009) | 0.6080 (±0.009) | **0.6698 (±0.009)** |
| **`kw_lift@3`** | **8.122 (±0.133)** | 5.585 (±0.091) | 4.216 (±0.064) | 3.619 (±0.050) |

- 两个 `lift` 仍然**严格单调**：降幅 2.96（嵌入口径）/ 4.50（纯词面口径），
  而各自极差只有 0.1 量级 → **结论 2（分块越细 lift 越高）站得住**。
- raw `chunk_precision@3` 仍非单调（最粗档 0.6759 最高）、raw `kw_precision@3` 仍单调递增 →
  再次确认 **raw precision 不可跨分块配置比较**，只能看 `lift`。

**检索器对比**（300/30、k=4、候选 10、3 次均值 ± 极差）：

| 指标 | vector | bm25 | hybrid |
|---|---|---|---|
| `doc_hit_rate@1` | 0.6204 (±0.0834) | 0.5000 (±0.0000) | 0.6204 (±0.0278) |
| `doc_mrr` | 0.7037 (±0.0695) | 0.6134 (±0.0000) | **0.7215** (±0.0231) |
| `kw_precision@1` | 0.5741 (±0.0555) | 0.5000 (±0.0000) | **0.6018** (±0.0278) |
| `kw_precision@3` | **0.5587** (±0.0092) | 0.5093 (±0.0000) | 0.5278 (±0.0185) |
| `chunk_precision@3` | **0.6666** (±0.0278) | 0.4167 (±0.0000) | 0.6111 (±0.0185) |

★ **BM25 三次运行的极差全部是 0.0000**（三次结果逐位相同）——
噪声完全来自向量侧，这是对"根因在 HNSW"的**独立佐证**。

按**噪声地板**与**口径质量**两个维度逐条复核（地板取该指标各报告极差的最大值）：

| 结论 | 口径 | 差距 | 地板 | 判定 |
|---|---|---|---|---|
| `kw_precision@3`（BM25 落后） | **对抗口径（无嵌入、无循环）** | 1.8 题 | 0.33 题 | ✅ **唯一能进结论区的独立证据** |
| `chunk_precision@3` 等 6 行（BM25 落后） | 弱标注（**嵌入同源、有循环**） | 9.0 题 | 1.0 题 | ⚠️ 只说明"**在嵌入语义视角下** BM25 排名靠后"，属**同源描述**，不算独立证据 |
| `kw_precision@1`（BM25 落后） | 对抗口径 | 2.7 题 | 2.0 题 | ❌ 不进结论区（余量 1.35× < 地板自身不确定度 1.5×） |
| `doc_hit@1`（BM25 落后） | 文档级 | 4.3 题 | 3.0 题 | ❌ 同上（余量 1.43×） |
| `doc_mrr`（vector vs hybrid） | 文档级 | 0.6 题 | 2.5 题 | ❌ 无法排序 |

**三层收口**（比上一轮更紧）：

1. **噪声层**：上一轮说"BM25 落后 5~6 题、地板 2 题、余量充足" —— "余量充足"收回。
2. **口径层（★ 这一层靠人，不靠机制）**：可分的那 8 行里，6 行是 `chunk_*`（正例由嵌入相似度定义，
   而 vector/hybrid 就是嵌入驱动）→ 对 BM25 **结构性不利**，这 6 行的 9 题差距
   **不能当"BM25 落后"的独立证据**。真正无循环的只有 `kw_*` 那 2 行（1.8 题）。
3. **估计层**：两个 `@1` 虽然越过了地板，但余量只有地板的 1.35~1.43 倍，
   而地板本身在小样本下可差 1.5 倍（4.11.4）→ 也**不进结论区**。

→ **结论区最终只剩一句**：在**不含嵌入**的对抗性口径下，BM25 的 `precision@3` 低 1.8 题
（地板 0.33 题）；`chunk_*` 上那些更大的差距只作为"同源视角下的描述"附注。

#### 4.11.4 地板本身也是估计值（别当成精确常数）

同一个 ck=10 配置：上一次用 **5 次**运行测到地板 0.0556（2 题），这次用 **3 次**测到 0.0834（3 题）。
两次都没算错 —— 极差是小样本估计，**样本越少越容易低估**（这次命中了更大的偏离样本）。

所以要读成"**≈3 题**"而不是"3.00 题"；要正式下结论建议 `--repeat 5`。

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

> 这份清单曾经腐化过：它把 README 骨架 / 真实嵌入 / IR 指标都写成"未做"，
> 而这三件**后来都做完了**（分别在 5.7、4.10~4.11、5.8）。现已按实际状态重写 ——
> **勾选状态必须能指到对应小节，否则它就是一份会说谎的清单。**

| 项 | 状态 | 见 |
|---|---|---|
| a) FastAPI 包装（:8020，`/health` + `/chat`，缺 Key 返回明确错误而非崩） | ✅ | 5.1 |
| b) 注册进 `svc.ps1`（短名 `kbapi`，BOM/CRLF/语法均校验通过） | ✅ | 5.1 |
| c) README 骨架（架构 + 检索策略 + 模块结构 + 缺陷清单 + 启动方式 + 测试与 CI） | ✅ | README |
| d) CHANGES 划定「有效产出 / 暂无效结论」两栏 | ✅ | 六 |
| 接入真实嵌入（`--embedding-backend dashscope\|local`，local 走本地 11434；按「后端 + 维度」隔离 collection） | ✅ | 5.6 / 5.7 |
| 建 IR 指标（Hit Rate@k / MRR / Recall@k / nDCG@k / lift；确定性，不需 LLM 裁判） | ✅ | 4.10 / 4.11 / README |
| **测试层 + CI**（98 条确定性单测；CI 只跑单测，不跑 IR/RAGAS） | ✅ | **5.8** |
| 重排的真实收益数字 | ⏳ **等 `DASHSCOPE_API_KEY`** | 4.7 |
| 真实 `report_optimized.json` 与真实对比表 | ⏳ 同上 | 3.6 / 4.7 |
| 本地 LLM 替换云端（零成本对比） | ⬜ 未做（非阻塞） | — |

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

### 5.8 测试层与 CI（原"无 CI、无单元测试"这个缺口已闭合）

#### 5.8.1 为什么现在才补

项目此前**没有 `.github`、没有 `tests/`**。评测层（IR 指标）替它承担了一部分"回归"职责，
但评测**走的是替身/独立路径**，恰恰发现不了 §5.2 那个致命缺陷：
`RagService` 完全无法构造（`ImportError`），而 mock 链路**从不 import `rag`**，
于是**评测全绿、系统其实一句都答不出来**。

§5.2 结尾已经把教训写成一句话："必须靠导入全量模块的冒烟测试兜住"。
本节就是把那句话变成**真正的测试**，并把 CI 建起来。

#### 5.8.2 98 条确定性测试（不联网 / 不需要 Key / 不需要本地嵌入服务）

| 测试文件 | 条数 | 钉住什么 |
|---|---|---|
| `tests/test_hygiene.py` | 8 | 配置路径必须绝对（换目录启动会读写到不同向量库）；**废弃配置不得复活**；代码里的 `CHANGES §x.y` 必须真有这一节；`.env` / 向量库 / 评测中间产物必须被 git 忽略 |
| `tests/test_chunking_and_extract.py` | 9 | ★ **chunk_size 对短文档失效**那个缺陷的回归（150 → 2 片段、1000 → 1 片段）；txt 的 utf-8/gbk 回退；PDF 走 pypdf；MD5 去重仍生效 |
| `tests/test_retrievers.py` | 16 | ★ 单路检索**必须仍截断到 k**（曾因短路优化返回 `candidate_k` 条，静默改变基线）；并列时排序由**内容指纹**决定（与返回顺序无关）；BM25 命中/过滤；重排无 Key 时降级不崩 |
| `tests/test_report_guardrails.py` | 12 | ★ 噪声地板取值口径 + **两级自动标注**（整行不可排序 / 仅首末噪声内）；四项口径拦截（`k` / `candidate_k` / 后端+维度 / 重排） |
| `tests/test_merge_reports.py` | 6 | ★ headline 噪声地板**不得被 `lift` 这类大量纲指标带跑偏**（曾算出"11.5 题 / 6"） |
| `tests/test_import_smoke.py` | 12 | ★ **§5.2 那个致命 ImportError 的固化回归**：全部模块能 import + `RagService` 能构造 + 正确导入路径有效 |
| `tests/test_ir_metrics.py` | 17 | `precision@k` 分母恒为 k；`lift` 换算；`base_rate` 不除零；三种口径文本必须自带"偏袒谁 / 有无循环性" |
| `tests/test_dataset.py` | 9 | 评测集结构校验；`answer_keywords` 必须真能在**来源文档**里找到（否则对抗口径**静默全 0**） |
| `tests/test_ci_summary.py` | 9 | CI summary 统计正确，且**"不跑 IR / 不跑 RAGAS"的声明不可腐化** |

本地跑：`python -m pytest` → **98 passed, 约 4 秒**。

#### 5.8.3 CI：跑什么、**不跑什么**（这是本次最容易做错的地方）

`.github/workflows/tests.yml`：Ubuntu + **Python 3.11**（必须：`chroma-hnswlib==0.7.3` 没有 cp312 轮子，
3.12 上要现场编译 C++ 并失败）、装三个 requirements、跑 pytest、上传 `junit.xml`、写 job summary。

★ **不跑什么，以及为什么必须写在 CI 页面上**（而不是只写在 workflow 注释里 —— 没人看注释）：

| 不跑 | 原因 |
|---|---|
| **IR 评测** | IR 指标本身确定，但**向量腿不是**（HNSW 近似检索，跨进程候选集合会变，§4.10.5 / §4.11.2）。做成门禁会**随机变红** |
| **RAGAS** | 4 个指标需要 LLM 裁判（`DASHSCOPE_API_KEY`），仓库里不存密钥 |

这段声明由 `scripts/ci_summary.py` 写进**每个 job 的 summary**，并且有 9 条测试钉住它 ——
**防止这句声明被删改之后，公开的绿勾被误读成"所有测试都过了"**。

#### 5.8.4 落地时抓到的两个事实（都不是猜的）

1. **`RagService` 构造时会真的发一次嵌入请求**：`rag.py` → `embeddings.prepare()` → `observed_dim()`
   （要实测向量维度）。所以"构造冒烟"**必须先换掉嵌入后端**，否则它会因 401 / 连不上本地服务而失败 ——
   那是**环境问题**，混进来会让这条最关键的测试变成"看运气"。
   （顺带修正一个理解：§5.2 的构造证据应当是在嵌入后端可用时取得的。）
2. **`BM25Okapi` 的 IDF 在小语料 / 高频词上会 ≤ 0**，而实现会把非正分片段过滤掉
   （"完全不相干的片段不要塞进来当噪声"）。所以 BM25 的测试语料必须够大、目标词必须够稀有，
   否则测出来是空列表 —— 那是**正常行为，不是 bug**。已写进测试注释，免得以后被人"修好"。

#### 5.8.5 命令

```powershell
# 装测试依赖（Windows 上必须先设 PYTHONUTF8，否则 pip 读带中文注释的文件会 GBK 解码错）
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -c evaluation\constraints-eval.txt

# 跑测试
.\.venv\Scripts\python.exe -m pytest

# 本地预览 CI 会写进页面的 summary
.\.venv\Scripts\python.exe -m pytest --junitxml=junit.xml
.\.venv\Scripts\python.exe scripts\ci_summary.py junit.xml
```

#### 5.8.6 仍未做的（如实列出）

- **覆盖率统计**：没装 `pytest-cov`。不追求覆盖率数字 —— 上面每一条都是"改动容易踩回去"的地方，
  不是为凑数。
- **端到端问答的自动化**：需要 `DASHSCOPE_API_KEY`，且与"IR 不进 CI"同一个理由（不稳定）。

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
| 12 | BM25 在文档级落后于 vector / hybrid（**扩语料后差距更明显**） | 小语料三个设置方向一致；12 篇文档上差距扩大到 18/36 vs 23/36 | 4.3-R / 4.9.6 |
| 13 | **两个口径陷阱的定量证据**：弱标注的循环性、以及弱标注**基准率随分块漂移** | 基准率实测从 9.0%（150/15）漂到 15.1%（1000/100），1.7 倍 | 4.9.4 |
| 14 | 扩语料与评测集构建**可复现**（12 篇 / 225,812 字 / 36 题；语义改写比例实测 38.9%） | 脚本可重跑；答案关键词校验拦下 2 个真错误 | 4.9.1 / 4.9.2 |
| 15 | **推翻自己**：上一轮的"分块越细噪声越少"（第 13 条旧版）在 12 篇文档上**未复现且反转**；连带的"结论强度排序"也被推翻（4.9.5） | 同一脚本、更大语料、结果相反 | 4.9.3 / 4.9.5 |
| 16 | **解释了第 15 条的反转**：raw `precision@k` 被基准率主导（基准率 9.01%→15.10%，而 precision 下界就是基准率）；归一化后的 `lift` 让两个**独立口径**（含无嵌入的对抗口径）一致严格单调 | 两口径同向且严格单调（7.708→4.476 / 8.033→3.602） | 4.10.2 |
| 17 | **"BM25 落后"经对抗性口径验证**：在一个**刻意不含嵌入**（因而无循环性）的口径下，BM25 依然最低 | 三口径同向：doc_hit@1 0.5000 vs 0.6389；kw_precision@3 0.5093 vs 0.5463；kw_lift@3 5.009 vs 5.373 | 4.10.3 |
| 18 | 新增两个口径纪律机制：`lift` 归一化（解决跨分组可比）+ **对抗性口径**（切断循环性）；拦截范围扩至 `k`/`candidate_k`/后端+维度/重排 | 机制存在、实测生效 | 4.10.4 |
| 19 | **实测出评测噪声地板**（向量腿跨进程不确定：`doc_hit@1` 2 题 / 36），并据此**逐条复核**已有结论哪些还在噪声之上 | 5 个独立进程重复实测；BM25 哈希跨进程一致、向量腿每次都变 | 4.10.5 |
| 20 | 新增 `--repeat N` 重复测量机制（**独立进程**，每配置各自测地板），报告层**自动标注**落在噪声内的对比行 | 检索器表 **39 行**中：5 行整行不可排序、26 行仅首末不可分、**8 行可分**（其中仅 2 行为无嵌入口径） | 4.11.1 / 4.11.3 |
| 21 | **负结果（有价值）**：确定性 tie-break 改造消不掉噪声 —— 改造后"仅顺序不同"= 0 题、"集合不同"= 24 题，证明只能披露 | 24/36 题集合级差异，且查询向量跨进程逐位一致 | 4.11.2 |
| 22 | 用 3 次均值刷新两张表，并把结论**三层收口**：噪声层（"余量充足"→分档）+ **口径层**（`chunk_*` 6 行是嵌入同源、不算独立证据，只有 `kw_*` 2 行能进结论区）+ **估计层**（两个 `@1` 余量 1.35~1.43× 小于地板自身 1.5× 不确定度，也不进） | 结论区最终只剩 1 句 | 4.11.3 |
| 23 | **报数纠错**：4.11.1 初版"35 行 / 4 行"是从**过滤后的终端输出**里数的（与 4.10.5 混入 `candidate_k=32` 属同类错误）；同时明确机制边界 —— 它筛噪声、**筛不了口径质量** | 逐行数源表：39 = 5 + 26 + 8 | 4.11.1 |
| 24 | **补齐测试层：98 条确定性单测**，其中包含 §5.2 那个**致命 ImportError** 的固化回归（此前只写在文档里的一句话命令）、`chunk_size` 缺陷回归、单路截断回归、噪声地板标注回归 | `python -m pytest` → 98 passed，约 4 秒 | 5.8.2 |
| 25 | **CI 建成，且把"不跑什么"写进 job summary**：IR 不进 CI（向量腿会随机变红）、RAGAS 不进 CI（需 Key）；这段声明由脚本生成并有 9 条测试钉住，防它腐化后被误读成"全都过了" | `.github/workflows/tests.yml` + `scripts/ci_summary.py` + `tests/test_ci_summary.py` | 5.8.3 |

这 25 条的共同点：**结论都配了可复现的证据，并且把局限写在结论旁边。**
（第 9 条是"能力验证"；第 12 条幅度在噪声内；第 15 条是**主动作废自己上一轮的结论**；
第 16、17 条是**用新口径把作废掉的结论重新做成可信的样子** —— 这一轮的价值不在数字变大，
而在"换了两种互不相同的标注机制，方向仍然一致"。）

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

7. `4.4-R` 的**原始形态**（小语料、未归一化的 raw `precision`）已作废 ——
   4.9.3 证明它在 12 篇文档上未复现且反转；我上一轮给的"结论强度排序"也被推翻（4.9.5）。
   **但 4.10 把它救回来了，只是换了更硬的形态**：引入 `lift` 归一化 + 对抗性口径后，
   两个独立口径一致严格单调（4.10.2）。

   写 README / 简历时，分块部分**只能**引用 4.10.2 的表述：
   > 在相对随机基线的口径下（`lift@3`），分块越细、top-k 里噪声占比越低，
   > 该趋势在"嵌入相似度"与"纯词面关键词"两种互不相同的标注口径下**同向且严格单调**。

   ❌ 仍然不能引用的：
   - "分块越细噪声越少"的 raw precision 版本（那就是被推翻的那个）；
   - 任何把 `chunk_*`（嵌入同源口径）单独当作公平对比的说法。

8. **凡是要下结论，必须先说自己测出的噪声地板**。三条关于地板本身的事实：
   - 它是**每个配置各自的属性**，不是全局常数：实测 ck=32 的四档是 0.3 / 1.0 / 0.3 / 1.0 题，ck=10 是 3.0 题（4.11.1）。
   - 它**本身是估计值**：同一配置 5 次运行测到 2 题、3 次运行测到 3 题。读成"≈3 题"，要正式下结论用 `--repeat 5`（4.11.4）。
   - 引用数字时以 **4.11.3 的 3 次均值表**为准（4.10.2 / 4.10.3 是单次运行，只作为噪声地板的证据保留）。
   具体到本文档：
   - ✅ 可以说（**唯一能进结论区的**）：在**不含嵌入**的对抗性口径下 BM25 的 `kw_precision@3` 低 1.8 题（地板 0.33 题）。
   - ✅ 可以说：**分块越细 lift 越高**（两个口径降幅 2.96 / 4.50，远大于极差 0.1 量级）。
   - ⚠️ 只能作**附注**：`chunk_*` 上 BM25 差 9 题 —— 但那个口径的正例由嵌入相似度定义，
     对 BM25 **结构性不利**，属"在嵌入语义视角下排名靠后"的**同源描述**，不是独立证据。
   - ❌ 不进结论区：`doc_hit@1`（4.3 题 vs 地板 3.0）与 `kw_precision@1`（2.7 题 vs 2.0）——
     余量只有地板的 1.35~1.43 倍，小于地板自身 1.5 倍的不确定度。
   - ❌ 不能说：**vector 与 hybrid 谁更好**（`doc_mrr` 差 0.6 题 vs 地板 2.5 题，无法排序）。
   - ❌ 不能依赖机制代替判断：**自动标注只筛噪声**（统计可比性），**筛不了口径质量** —— 循环性不是统计问题。

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
