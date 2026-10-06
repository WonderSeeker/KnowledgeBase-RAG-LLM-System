# 改造记录（CHANGES）

> 记录本仓库的改造过程：原项目是什么 / 我改了什么 / 量化结果。
> 每个阶段一条，量化数据全部来自本机实测。

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

### 4.3 三种检索策略实测（⚠️ 离线启发式，不是 RAGAS）

同一套 29 条评测集，只改检索策略：

| 指标 | vector（top_k=1） | bm25（top_k=4） | hybrid（top_k=4，候选 6） |
|---|---|---|---|
| faithfulness | 0.9655 | 0.9310 | 0.9655 |
| answer_relevancy | 0.2146 | 0.2105 | 0.2146 |
| **context_recall** | 0.6972 | 0.7159 | **0.7481** |
| context_precision | 0.6972 | 0.4793 | 0.4313 |
| 平均检索片段数 | 0.97 | — | 3.0 |

混合检索的召回高于任一路单独使用，符合 RRF 的设计预期。

### 4.4 分块参数对比实验（⚠️ 离线启发式，不是 RAGAS）

`evaluation/experiments.py --mode mock --configs 300/30,600/60,1000/100 --retriever hybrid --top-k 4 --candidate-k 12`

| 指标 | 300/30 | 600/60 | 1000/100 |
|---|---|---|---|
| 语料片段总数（`--dry-run` 实测） | 15 | 9 | 6 |
| context_recall | 0.7210 | 0.7439 | **0.7481** |
| context_precision | 0.3296 | 0.3774 | **0.4313** |
| answer_relevancy | **0.2875** | 0.2501 | 0.2146 |

分块越小 → 片段越聚焦、答案相关性越高，但召回略降；分块越大 → 召回略升，但噪声更多。
这是一条真实的权衡曲线，不是"某组参数一定最好"，需要按业务取舍。

### 4.5 这一步挖出来的 4 个真问题

1. **`max_spliter_char_number = 1000` 让"调小 chunk_size"对短文档完全无效**
   `尺码推荐.txt` 只有 270 字，无论 `chunk_size` 设 300 还是 1500，永远是 1 个片段（`--dry-run` 实测）。
   也就是说这份最需要精确检索的尺码表，永远无法被拆开检索。
   → 阶段 4 建议把这套"先判长度再切分"的逻辑收敛成单纯的 splitter。
2. **语义改写类问题是字面检索的死角**
   `color-06「怎样穿显得更瘦？」` 在离线替身下 **0 命中**——语料里写的是"显瘦搭配"，
   字面/BM25 都匹配不上，只有真向量检索能救。这条样本正好是验证向量检索价值的标杆。
3. **基线的 `top_k=1` 有 3% 的问题直接 0 命中**（平均片段数 0.97），等于大模型收到"无相关参考资料"。
4. **Chroma 匿名遥测在 stderr 刷错误日志**（新版 posthog 与 chromadb 0.4.x 不兼容），
   已通过 `Settings(anonymized_telemetry=False)` 关闭。

### 4.6 命令

```powershell
cd D:\Projects\KnowledgeBase-RAG-LLM-System

# 不联网看清分块效果（不需要 Key）
.\.venv\Scripts\python.exe scripts\ingest.py --dry-run --chunk-size 300 --chunk-overlap 30

# 离线跑通实验流程（分块真实生效，检索与打分用离线替身）
.\.venv\Scripts\python.exe evaluation\experiments.py --mode mock --configs 300/30,600/60,1000/100 --retriever hybrid --top-k 4

# 真实实验（需要 Key）：每组分块参数入独立集合，再用 RAGAS 打分
.\.venv\Scripts\python.exe evaluation\experiments.py --mode real --name chunk --configs 500/50,1000/100,1500/200 --retriever hybrid --top-k 4 --candidate-k 10 --rerank

# 基线 vs 优化 的真实对比表
.\.venv\Scripts\python.exe evaluation\report.py evaluation\report_baseline.json evaluation\report_optimized.json --out evaluation\REPORT.md
```

### 4.7 阶段 3 验收状态

- [x] BM25 + 向量混合检索（RRF 融合）
- [x] Rerank 接入（DashScope `gte-rerank-v2`），失败降级已实测
- [x] 分块大小/重叠度对比实验流程（`--dry-run` 试算 + 多组评测 + 横向表）
- [x] 「改造前 vs 改造后」对比表能力（`report.py` 支持 1/N 份报告）
- [ ] **真实 `report_optimized.json` 与真实对比表** —— 阻塞在 `DASHSCOPE_API_KEY`
- [ ] 重排的真实收益数字 —— 需要 Key 才能测

---

## 五、阶段 4

（未开始）

---

## 六、禁止事项（改造红线，勿破）

1. 不重建 `.venv`，不换成 Python 3.12（`chroma-hnswlib==0.7.3` 在 3.12 上只能现场编译 C++，会失败）。
2. 不改 `requirements.txt` 的版本锁定。
3. `requirements.txt` 含中文注释，直接 pip 装会 `UnicodeDecodeError: 'gbk'`，需先 `$env:PYTHONUTF8='1'`。
4. `DASHSCOPE_API_KEY` 只放 `.env`，禁止写进代码或提交。
5. 不下载任何 GitHub Release 文件（通道被阻断）。
