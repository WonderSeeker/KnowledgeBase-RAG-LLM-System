# KnowledgeBase RAG LLM System

在本地上传文档 → 切分入库 → 检索增强问答，并**带一套评测层把"效果好坏"量化出来**的工程化 RAG 项目。

---

## 项目状态

**已完成并验证**（本仓库可复现的证据）：

- 检索层：向量 / BM25 / RRF 混合 / 重排，统一 `invoke()` 契约 —— 8 项探针通过（CHANGES §6.1）
- 修复 4 个上游缺陷（含一个使问答链路完全不可用的 ImportError，以及两个**静默型**缺陷），
  并定位修掉 1 处自己在改造中引入的静默回归（CHANGES §2.1 / §4.8 / §5.2 / §5.7）
- 2 个机制性护栏：评测口径不一致拦截、遥测治理（CHANGES §4.3 / §5.3）
- FastAPI 服务（`/health` + `/chat`），缺 Key 时返回明确错误而非 500（CHANGES §5.1）

**进行中**：效果对比数据（检索策略 / 分块参数）待接入真实嵌入后产出。
在此之前，`CHANGES.md` 中标注「离线启发式」的表格**不作为结论**，理由见其第六节。

---

## 这是什么 / 解决什么问题

上游项目是一个能跑通的 RAG demo：上传 txt、切分、写 Chroma、检索、让通义千问作答。
它**缺的是工程化闭环**——没有评测，效果好坏只能靠肉眼看；没有 API，只能开浏览器；没有检索优化。

本仓库在原项目基础上补齐三件事：

1. **能证明效果**：一套 RAGAS 评测层（29 条中文评测集 + 4 指标 + 报告/对比表），
   以及一条把「入库 → 评测 → 出横向表」串起来的对比实验命令；
2. **能提升效果**：检索层从"纯向量 + `k=1`"扩展为 BM25 / RRF 混合 / 重排可选，
   分块参数可实验、可复现；
3. **能对外服务**：FastAPI 暴露 `/health` 与 `/chat`，并统一用 `svc.ps1` 管理启停。

同时修掉了 3 个真实缺陷，其中一个是**使问答链路完全不可用**的 `ImportError`（见下文第 6 节）。

### 效果示例

<div align="left">
  <img src="./assets/chat_demo1.png" width="760" alt="Streamlit 问答页：基于知识库尺码表作答，并说明该问题超出参考文档覆盖范围">
</div>

问「我的体重 990 公斤，按照参考文档推荐尺码」时，模型**没有编造尺码**，而是引用知识库里的尺码表
（最大 5XL / 190cm 以上 / 210 斤以上），并明确指出该问题已超出参考文档的覆盖范围 ——
这正是 RAG 该有的行为：有据可依，且知道自己不知道。

> 截图为原项目 demo，本项目的问答页流程与示例语料一致（仅额外增加了一行耗时显示）。

---

## 架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ① 接入层                                                                  │
│                                                                          │
│   Streamlit UI            app_chat.py    :8501   svc.ps1 短名  rag       │
│                           app_upload.py          （上传页，未注册短名）   │
│                                                                          │
│   FastAPI 服务            server.py      :8020   svc.ps1 短名  kbapi     │
│                             GET  /health   健康检查（不需要 Key）          │
│                             GET  /config   当前检索/分块/模型配置          │
│                             POST /chat     问答                           │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │  RagService.answer(question)
┌────────────────────────────────▼─────────────────────────────────────────┐
│ ② RAG 编排层              rag.py                                          │
│                                                                          │
│   retrieve() → 检索器 → format_document → PromptTemplate → ChatModel     │
│   RunnableWithMessageHistory ── file_history_store.py （多轮历史落盘）     │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │  candidate_k 个候选片段
┌────────────────────────────────▼─────────────────────────────────────────┐
│ ③ 检索层                  retrievers.py                                   │
│                                                                          │
│   VectorRetriever ─┐                                                     │
│   BM25Retriever  ──┼─→ RRF 融合 ─→（可选）DashScopeReranker ─→ top_k      │
│                    └─→ HybridRetriever（三者统一 invoke 契约）            │
│                                                                          │
│   ★ 嵌入接入点：config_data.embedding_model_name                          │
│       当前：DashScope text-embedding-v4（云端，需 DASHSCOPE_API_KEY）      │
│       待接入：本地 http://127.0.0.1:11434/v1/embeddings（bge-small-zh）    │
│       ⚠️ 不同嵌入后端维度不同，必须按「后端 + 维度」隔离 collection         │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│ ④ 存储        Chroma 持久化于 chroma_db/     分块参数与集合名均可实验隔离   │
└──────────────────────────────────────────────────────────────────────────┘
```

模块职责（一个文件一件事）：

| 文件 | 职责 |
|---|---|
| `server.py` | FastAPI 接口层（`/health` `/config` `/chat`），含错误语义与懒加载 |
| `app_chat.py` / `app_upload.py` | Streamlit 问答页 / 上传页（原项目入口，保留） |
| `rag.py` | RAG 链路编排：检索 → 提示词 → 生成 → 多轮历史 |
| `retrievers.py` | 检索策略层：向量 / BM25 / RRF 混合 / 重排 |
| `vector_stores.py` | Chroma 封装（集合名、`k` 覆盖、遥测设置） |
| `knowledge_base.py` | 文档抽取（txt / pdf）、切分、MD5 去重、入库 |
| `file_history_store.py` | 会话历史文件存储 |
| `config_data.py` | 全局配置：模型、分块、检索策略、路径 |
| `evaluation/` | 评测层：评测集、RAGAS 执行、报告、对比实验驱动 |
| `scripts/ingest.py` | 命令行入库（支持分块参数与集合隔离、`--dry-run` 试算） |

---

## 快速开始

> 启停统一走 `D:\Projects\svc.ps1`（它同时约定了端口、进程隔离与日志位置），不提供第二套启动方式。

### 1. 配置 API Key

`/health`、`/config` 不需要 Key；**只有 `/chat` 与查询需要**。把 Key 写入项目根目录的 `.env`：

```
DASHSCOPE_API_KEY=sk-xxxxxxxx
```

Key 申请：<https://dashscope.console.aliyun.com/apiKey>。`.env` 已在 `.gitignore` 中排除，不会被提交。

> **可选：把嵌入切到本地（零成本）。** 默认嵌入走 DashScope 云端；若本机有 OpenAI 兼容的嵌入服务
> （`http://127.0.0.1:11434`，模型 `bge-small-zh-v1.5` / 512 维，用 `.\svc.ps1 start llm` 启动），
> 可切到本地：入库用 `scripts\ingest.py --embedding-backend local --rebuild`，评测侧加同样的参数。
> 两个后端的**向量维度不同，绝不能共用同一个集合** —— 默认已按后端隔离为 `rag` / `rag_local`，
> 且代码会把「后端 + 模型 + 实测维度」写进集合元数据，不一致直接报错退出。
> 注意：**生成模型仍走通义千问，所以 `/chat` 依然需要 Key**。

### 2. 启停服务

```powershell
cd D:\Projects

.\svc.ps1 list              # 查看全部服务与状态
.\svc.ps1 start kbapi       # 启动 FastAPI 接口 -> http://127.0.0.1:8020
.\svc.ps1 start rag         # 启动 Streamlit 问答页 -> http://127.0.0.1:8501
.\svc.ps1 health            # 对所有已运行服务做一次健康探测
.\svc.ps1 log kbapi         # 实时看日志
.\svc.ps1 stop kbapi        # 用完即停（本机内存只有 15.2 GB）
```

日志位置：`D:\Projects\_logs\kbapi.out` / `.err`。

### 3. 先确认服务状态

```powershell
curl.exe http://127.0.0.1:8020/health
```

返回 `status` 为 `ok` 表示可以问答；`degraded` 时 `reasons` 会写明缺什么（缺 Key / 语料为空）。

### 4. 入库语料

入库走命令行脚本（可复现、可换分块参数；`--dry-run` 不联网也能先看切分结果）：

```powershell
cd D:\Projects\KnowledgeBase-RAG-LLM-System
.\.venv\Scripts\python.exe scripts\ingest.py                              # 默认入库 assets/ 下全部 txt
.\.venv\Scripts\python.exe scripts\ingest.py --dry-run --chunk-size 300 --chunk-overlap 30
```

### 5. 提问

```powershell
curl.exe -s -X POST http://127.0.0.1:8020/chat -H "Content-Type: application/json" `
  --data-binary "@body.json"
```

`body.json`（UTF-8）：

```json
{"question": "身高168cm、体重100斤，建议穿什么尺码？", "session_id": "demo"}
```

浏览器交互式文档：<http://127.0.0.1:8020/docs>

---

## 接口

### `GET /health`

| 字段 | 含义 |
|---|---|
| `status` | `ok` = 可以问答；`degraded` = 进程活着但还不能问答 |
| `reasons` | `degraded` 的原因（缺 Key / 语料为空），`ok` 时为空数组 |
| `api_key_configured` | `DASHSCOPE_API_KEY` 是否已配置 |
| `collection` | 当前检索的 Chroma 集合名 |
| `documents` | 向量库里的片段数；`null` 表示读取失败（不影响进程状态） |
| `retrieval` | 当前检索配置：`method` / `top_k` / `candidate_k` / `rrf_k` / `rerank` / `rerank_model` / `collection` |
| `chunking` | `chunk_size` / `chunk_overlap` |
| `models` | `embedding` / `chat` 模型名，以及 `embedding_backend` / `embedding_expected_dim`（后者是配置声明的期望值；实测维度要在首次真实调用后才确定，不在 `/health` 里冒充） |
| `version` | 接口版本 |

### `GET /config`

| 字段 | 含义 |
|---|---|
| `retrieval` | 与 `/health.retrieval` 同结构（`method` / `top_k` / `candidate_k` / `rrf_k` / `rerank` / `rerank_model` / `collection`） |
| `chunking` | `chunk_size` / `chunk_overlap` |
| `models` | 同 `/health.models`（含 `embedding_backend` / `embedding_expected_dim`） |
| `paths` | `persist_directory`（向量库目录）/ `md5_path`（入库去重记录） |
| `version` | 接口版本 |

### `POST /chat`

请求：`{"question": "...", "session_id": "..."}`（`question` 必填，长度 1~2000；`session_id` 用于多轮历史，默认 `api_user`，最长 64）

响应：`answer`、`contexts`（命中的片段原文，便于核对依据）、`n_contexts`、`session_id`、
`retrieve_seconds`、`generate_seconds`、`total_seconds`、`retrieval`（本次实际使用的检索配置）。

### 错误语义

设计原则：**不返回 500 堆栈**，每种失败都给出可操作的下一步。下表每一行都经实机验证：

| HTTP | `error` | 触发条件 | 返回的提示 | 验证方式 |
|---|---|---|---|---|
| 503 | `api_key_missing` | 未配置 `DASHSCOPE_API_KEY` | 把 Key 填进 `.env` 后 `.\svc.ps1 restart kbapi` | 实机 HTTP 请求 |
| 503 | `knowledge_base_empty` | 向量库片段数为 0 | 先跑 `scripts\ingest.py` 入库 | handler 级调用（mock 片段数=0） |
| 502 | `upstream_error` | 模型服务调用失败 | 检查 Key 有效性/网络，或看 `_logs\kbapi.out`，并附原始异常 | handler 级调用（注入上游异常） |
| 422 | — | 参数校验失败（空问题 / 超 2000 字） | Pydantic 标准结构 | 实机 HTTP 请求（空 + 2001 字各一次） |

> 门禁顺序是设计要点：`/chat` **先判断 Key，再 `import rag`**。
> 因为 `DashScopeEmbeddings` 在**构造时**就校验 Key，顺序反了会变成 500 堆栈。

---

## 我修复的缺陷

只列两个最有分量的（完整清单与证据见 `CHANGES.md`）：

| 缺陷 | 后果 | 为什么不是偶发 |
|---|---|---|
| `langchain-core` 0.1.53 不导出 `RunnableWithMessageHistory` | **问答链路完全不可用**：`import rag` 直接 `ImportError`，`RagService` 无法构造 | 项目自声明的依赖范围 `>=0.1.52,<0.2.0` 内就会解析到 0.1.53，在该范围内**必然发生**，属版本适配缺陷 |
| `chunk_size` 对短文档失效（`max_spliter_char_number` 硬门槛） | 短文档只切出 1 个片段，检索退化为"全文匹配" | 客观对照：`chunk_size=150` 下片段数 **1 → 2**（CHANGES §4.8） |

另外 3 个（详细定位与证据见 `CHANGES.md` §2.1 / §5.7）：

- **`.env` 从未被加载**：依赖里装了 `python-dotenv`，但全项目 `load_dotenv()` 调用数为 0 ——
  按文档配置 `.env` 的用户拿不到 Key。
- **`metadatas` 拼成 `metadata`（静默型）**：LangChain 的参数名是复数，写单数会落进 `**kwargs` 被静默忽略，
  导致**所有片段元数据丢失**（实测 6/6 为 `None`），来源归因与文档级指标都不可能。
- **Chroma 持久化被关掉（静默型回归，我在改造中引入的）**：为关遥测传入了 `client_settings`，
  使 langchain 不再补 `is_persistent=True`，退化成**进程内内存库** ——
  入库进程自测 100% 通过，只有跨进程才暴露（CHANGES §5.7 有修复前后的对照输出）。

> 本节的缺陷清单只讲事实，不含任何效果数字。

---

## 评测框架与口径纪律

`evaluation/` 提供一条命令跑出评测报告：

```powershell
.\.venv\Scripts\python.exe evaluation\run_ragas.py --mode real --name baseline                  # RAGAS 4 指标
.\.venv\Scripts\python.exe evaluation\experiments.py --mode real --name chunk `
    --configs 500/50,1000/100,1500/200 --retriever hybrid --top-k 4 --candidate-k 10           # 对比实验
.\.venv\Scripts\python.exe evaluation\report.py evaluation\report_baseline.json --out evaluation\REPORT.md
```

4 个指标：`faithfulness` / `answer_relevancy` / `context_recall` / `context_precision`。
评测集 29 条中文样本（`evaluation/eval_set.jsonl`），覆盖精确数值、单点事实、跨段落归纳、否定性提问。

**口径纪律**（这是本项目刻意立的规矩，不是免责声明）：

1. **同名指标在不同口径下算法完全不同**。报告里每条指标都逐条写明当次口径的实际算法
   （离线启发式 = 字符二元组重叠率；RAGAS = 裁判 LLM 判定），并在 `meta` 中写死 `mode` 与 `judge`。
2. **`--mode mock` 的数字不是结论**。mock 模式下检索腿与打分器都是离线替身，
   两条替身相乘的结果不具备可迁移性。
3. **比对前先对齐口径**。报告层会主动拦截 `k` 不一致的对比；
   接入本地嵌入后，拦截条件扩展为「`k`、嵌入后端、维度任一不一致」。
4. 哪些产出可以作为结论、哪些必须等真实嵌入复测，见 **`CHANGES.md` 第六节「结论有效性划定」**。

> 检索策略与分块参数的效果对比：见下方「效果结论」。RAGAS 4 指标的真实数值仍需可用的云端 Key。

### 效果结论（真实嵌入 512 维 + 确定性 IR 指标）

评测条件：语料为 `corpus/` 下 **12 篇 Python 中文文档（225,812 字）**，
评测集 **36 题**（`evaluation/eval_set_v2.jsonl`，**实测 38.9% 为语义改写型**问题）。
完整数据、口径陷阱、噪声地板与复现命令见 `CHANGES.md` §4.9 ~ §4.11。

#### 三种标注口径（缺一个都会误读）

| 前缀 | 正例怎么定 | 含嵌入？ | 循环性 | 已知偏袒 |
|---|---|---|---|---|
| `doc_` | 评测集标注的应命中文档 | 否 | 无 | — |
| `kw_` | 片段文本含该题的 `answer_keywords`（**纯词面**） | **否** | **无** | 可能偏袒关键词检索 |
| `chunk_` | 与标准答案嵌入相似度 ≥ 0.6 | 是 | **有**（与向量检索同源） | 向量 / 混合 |

#### ★ 噪声地板：任何结论都必须先过这一关

实测发现 **chromadb 的向量检索跨进程不确定**：HNSW 近似检索、索引图在进程内重建，
同一配置起独立进程重跑，**36 题里有 24 题的候选集合本身就不同**（不只是顺序）。
因此同一配置的指标会漂：`doc_hit@1` 实测抖动 **≈3 题 / 36**；而 **BM25 三次运行的极差是 0.0000**（完全确定）。

所以本项目：
- IR 评测默认 `--repeat 3`（起 3 个**独立进程**取均值），并记录**该配置**的实测噪声地板；
- 报告层**自动**把落在噪声地板内的对比行标注为"无法排序" —— 不靠人去记得。

**下面每条结论都标注了它与噪声地板的关系**（地板是每配置各自的属性，且本身是小样本估计：
同一配置 5 次运行测到 2 题、3 次运行测到 3 题，故读作"≈3 题"）。
证据与机制见 `CHANGES.md` §4.10.5 与 §4.11。

#### 结论 1（成立，但**只有一条口径能当独立证据**）：BM25 落后

固定分块 300/30、`k=4`、候选 10、3 次独立进程均值（括号内为该指标实测极差）：

| 指标 | 口径性质 | vector | bm25 | hybrid |
|---|---|---|---|---|
| `kw_precision@3` | **对抗口径（无嵌入、无循环）** | **0.5587** (±0.009) | 0.5093 (±0.000) | 0.5278 (±0.019) |
| `kw_precision@1` | 对抗口径 | 0.5741 (±0.056) | 0.5000 (±0.000) | **0.6018** (±0.028) |
| `doc_hit@1` | 文档级（无循环） | 0.6204 (±0.083) | 0.5000 (±0.000) | 0.6204 (±0.028) |
| `chunk_precision@3` | 弱标注（**嵌入同源，对 BM25 结构性不利**） | 0.6666 (±0.028) | 0.4167 (±0.000) | 0.6111 (±0.019) |

**唯一能从这张表进结论的**：

> 在**不含嵌入**的对抗性口径下，BM25 的 `kw_precision@3` 低 **1.8 题**（地板 **0.33 题**）。

其余三行为什么不能进结论区：

| 行 | 差距 / 地板 | 为什么不用 |
|---|---|---|
| `chunk_precision@3` | 9.0 题 / 1.0 题 | 差距虽最大，但该口径的正例由嵌入相似度定义 —— 拿它下结论属于**同源描述**（用一个本就不利于 BM25 的口径去证明 BM25 不行），不是独立证据 |
| `doc_hit@1` | 4.3 题 / 3.0 题 | 余量 1.43× < 地板自身 1.5× 的不确定度（地板是小样本估计） |
| `kw_precision@1` | 2.7 题 / 2.0 题 | 同上（1.35×） |

机制一致：BM25 的短板集中在语义改写类问题（问题与原文无共同关键词时字面匹配直接漏检）。

#### 结论 2（成立）：分块越细，top-k 里噪声占比越低

用 `lift@k = precision@k / 基准率` 归一化后（原始 `precision` 因基准率漂移不可跨配置比较）：

| `lift@3`（3 次均值 ± 极差） | 150/15（1941 片段） | 300/30（872） | 600/60（422） | 1000/100（252） |
|---|---|---|---|---|
| 嵌入相似度口径 | **7.435 (±0.102)** | 6.278 (±0.092) | 4.873 (±0.000) | 4.476 (±0.000) |
| 纯词面关键词口径（无嵌入） | **8.122 (±0.133)** | 5.585 (±0.091) | 4.216 (±0.064) | 3.619 (±0.050) |

两个**标注机制完全不同**的口径给出**同向且严格单调**的结果，
降幅分别是 2.96 与 4.50，**远大于各自极差（0.1 量级）**。
（顺带：同一张表里 raw `chunk_precision@3` 仍非单调、raw `kw_precision@3` 仍单调递增 ——
方向相反的两组 raw 数字，正好说明为什么必须用 `lift`。）

#### ❌ 不能说：vector 与 hybrid 谁更好

`doc_hit@1` 两者并列（0.6204），`doc_mrr` 差 0.6 题、
**落在 2.5 题噪声地板之内 —— 无法排序**。
同理，"混合检索必然更好"在这份语料上得不到支持。

#### 一个被推翻、又被重建的结论（留痕，避免误引用）

早期我在 3 篇小语料上得到过"分块越细、top-k 噪声越少（四档单调）"。
换到 12 篇文档 + 36 题后**原始形态未复现，甚至反转**：

| raw `chunk_precision@3` | 150/15 | 300/30 | 600/60 | 1000/100 |
|---|---|---|---|---|
| 3 次均值 | 0.6698 | 0.6327 | 0.6111 | **0.6759** |

原因：弱标注的**基准率随分块变粗而漂移**（正例占语料比 9.01% → 15.10%），
而 `precision@k` 的**下界就是基准率** —— 越粗的两档"随机也能蒙对"的概率更高。
**所以 raw `chunk_precision` 的版本已撤下。**

它后来被 `lift` 归一化 + 第二个（无嵌入）口径重新建成结论 2（见上）——
**但要引用的是 lift 版本，不是 raw precision 版本。**

#### 三个必须知道的口径陷阱

1. **弱标注的循环性**：`chunk_*` 的正例由「与标准答案的嵌入相似度」定义，
   而 vector/hybrid 本身就是嵌入驱动 —— 用嵌入定义相关性再测嵌入检索，**结构性地偏袒它们**。
   → 所以才需要 `kw_`（纯词面、无嵌入）这个对抗性口径来交叉验证。
2. **弱标注的基准率会随分组漂移**：因此 **`chunk_*` / `kw_*` 的 raw precision 不可跨分块配置比较**，
   要比就比 `lift@k`。
3. **噪声地板（最容易忽略的一个）**：向量腿跨进程不确定，`doc_hit@1` ±≈3 题 / 36。
   **凡是要下结论，先看差距有没有超过它**；报告层已把它做成自动标注，但仍然要知道它的存在。

#### 其余局限

- **IR 指标不要放进 CI 门禁**：向量腿跨进程不确定，做成门禁会随机变红。
  本项目的 CI **只跑确定性单元测试**（`.github/workflows/tests.yml`），
  **不跑 IR 评测、也不跑 RAGAS** —— 每个 job 的 summary 里都有这句声明（见下节）。
- `candidate_k` 也是隐藏变量：同一配置（hybrid + 300/30）**同为 3 次均值**时，
  候选 32 的 `doc_hit@1 = 0.6759` (±0.028)、候选 10 的 `doc_hit@1 = 0.6204` (±0.028)。
  差 2 题、地板 1 题 —— 余量仅 2× 地板，**属边界，不作为结论**。
  **跨报告比较前必须同时对齐 `k` 与 `candidate_k`**（报告层已拦截 `k`/`candidate_k`/后端+维度/重排四项不一致）。
- `recall@k` 不可跨分块配置比较（语料片段数不同）。
- 噪声地板本身是小样本估计（同配置 5 次测到 2 题、3 次测到 3 题），读作"≈3 题"；正式下结论建议 `--repeat 5`。
- 评测集由 AI 撰写（非人工标注），构造原则、实测改写比例（38.9%）与校验方式见 `CHANGES.md` §4.9.2。

---

## 测试与 CI

```powershell
# 装测试依赖（Windows 上必须先设 PYTHONUTF8，否则 pip 读带中文注释的文件会 GBK 解码错）
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# 跑测试（98 条，约 4 秒；不联网、不需要 API Key、不需要本地嵌入服务）
.\.venv\Scripts\python.exe -m pytest

# 生成 job summary 并预览（CI 里靠 GITHUB_STEP_SUMMARY 追加到页面上）
.\.venv\Scripts\python.exe -m pytest --junitxml=junit.xml
.\.venv\Scripts\python.exe scripts\ci_summary.py junit.xml
```

**测什么**（都是"改动容易踩回去"的地方，不是为覆盖率凑数）：

| 测试文件 | 钉住什么 |
|---|---|
| `test_hygiene.py` | 配置路径必须绝对（换目录启动会读写到不同向量库）；**废弃配置不得复活**；代码里的 `CHANGES §x.y` 必须真的有这一节；`.env` 必须被 git 忽略 |
| `test_chunking_and_extract.py` | ★ **chunk_size 对短文档失效**那个缺陷的回归（150 → 2 片段、1000 → 1 片段）；txt 的 utf-8/gbk 回退；PDF 走 pypdf；MD5 去重仍生效 |
| `test_retrievers.py` | ★ 单路检索**必须仍截断到 k**（曾因短路优化返回 candidate_k 条，静默改变基线）；并列时排序由**内容指纹**决定（与返回顺序无关）；重排无 Key 时降级不崩 |
| `test_report_guardrails.py` | ★ 噪声地板取值口径 + **两级自动标注**（整行不可排序 / 仅首末噪声内）；四项口径拦截（k / candidate_k / 后端+维度 / 重排） |
| `test_merge_reports.py` | ★ headline 噪声地板**不得被 lift 这类大量纲指标带跑偏**（曾算出"11.5 题 / 6"） |
| `test_ir_metrics.py` | `precision@k` 的分母恒为 k；`lift` 换算；`base_rate` 不除零；三种口径文本必须自带"偏袒谁/有无循环性" |
| `test_dataset.py` | 评测集结构校验；`answer_keywords` 必须真的能在**来源文档**里找到（否则对抗口径静默全 0） |
| `test_import_smoke.py` | ★ 全部模块能 import + **`RagService` 能构造** —— 这是 `CHANGES` §5.2 那个**致命 ImportError** 的固化回归（当时它只是文档里的一句命令，而 mock 评测链路发现不了它） |
| `test_ci_summary.py` | CI summary 的统计正确，且**"不跑 IR / 不跑 RAGAS"的声明不可腐化** |

**CI 刻意不跑什么**（`.github/workflows/tests.yml`，每个 job 的 summary 都会打印这段声明）：

- ❌ **不跑 IR 评测** —— IR 指标本身确定，但**向量腿不是**（HNSW 近似检索，跨进程候选集合会变），
  做成门禁会随机变红。它只在需要时人工跑，且必须带 `--repeat 3` 输出实测噪声地板。
- ❌ **不跑 RAGAS** —— 4 个指标需要 LLM 裁判（`DASHSCOPE_API_KEY`），仓库里不存密钥。

---

## 目录结构

```text
KnowledgeBase-RAG-LLM-System/
├─ server.py                  FastAPI 接口层（:8020）
├─ app_chat.py                Streamlit 问答页（:8501）
├─ app_upload.py              Streamlit 上传页
├─ rag.py                     RAG 链路编排（检索 → 提示词 → 生成）
├─ retrievers.py              检索策略层（向量 / BM25 / RRF / 重排）
├─ vector_stores.py           Chroma 封装
├─ knowledge_base.py          文档抽取（txt/pdf）、切分、去重、入库
├─ embeddings.py              嵌入后端（dashscope / local）与向量库指纹校验
├─ file_history_store.py      会话历史文件存储
├─ config_data.py             全局配置
├─ .github/workflows/
│  └─ tests.yml               CI：只跑确定性单测（不跑 IR、不跑 RAGAS）
├─ tests/                     ★ 98 条确定性单测（见上节）
│  ├─ conftest.py             公共装置（路径注入 / 状态隔离 / 离线嵌入替身）
│  ├─ test_hygiene.py         路径、废弃配置、文档交叉引用、密钥忽略
│  ├─ test_chunking_and_extract.py
│  ├─ test_retrievers.py
│  ├─ test_report_guardrails.py
│  ├─ test_merge_reports.py
│  ├─ test_import_smoke.py    ★ 全模块 import + RagService 可构造（§5.2 缺陷回归）
│  ├─ test_ir_metrics.py
│  ├─ test_dataset.py
│  └─ test_ci_summary.py
├─ scripts/
│  ├─ ingest.py               命令行入库（支持分块参数 / 集合隔离 / --dry-run）
│  ├─ build_corpus.py         抓取评测语料（可重跑）
│  └─ ci_summary.py           把 junit.xml 变成 job summary（含覆盖范围声明）
├─ evaluation/
│  ├─ dataset.py              评测集构建（29 条内置样本 + JSONL 读写）
│  ├─ eval_set.jsonl          权威评测集（可直接手工增删）
│  ├─ eval_set_v2.jsonl       扩语料后的评测集（36 题）
│  ├─ build_eval_set.py       出题脚本（含答案关键词程序化校验）
│  ├─ ir_metrics.py           确定性 IR 指标 + lift 归一化 + 口径文本
│  ├─ run_ir.py               跑 IR 评测（支持 --repeat N 输出噪声地板）
│  ├─ run_ragas.py            跑评测：链路产出 + RAGAS 打分 + 落 JSON
│  ├─ offline.py              离线替身（本地切分 / 字面检索 / 启发式代理指标）
│  ├─ report.py               报告与对比表（含指标口径说明 + 噪声内自动标注）
│  ├─ experiments.py          对比实验驱动（分块参数 × 检索策略）
│  ├─ requirements-eval.txt   评测依赖（ragas 0.1.21）
│  └─ constraints-eval.txt    主依赖版本约束（防误升级）
├─ requirements.txt           原项目依赖（版本锁定未改动）
├─ requirements-optim.txt     检索优化依赖（rank-bm25 + jieba）
├─ requirements-dev.txt       测试依赖（pytest）
├─ pytest.ini
├─ corpus/                    评测语料（12 篇 Python 中文文档）
├─ assets/                    示例语料与 README 配图
├─ CHANGES.md                 改造记录：原项目 / 改了什么 / 量化结果 / 结论有效性划定
└─ LICENSE
```

---

## 许可与致谢

- 本项目基于 [lhh737/KnowledgeBase-RAG-LLM-System](https://github.com/lhh737/KnowledgeBase-RAG-LLM-System)（MIT）改造，遵循 MIT 许可。
- 致谢：Streamlit、LangChain、Chroma、阿里云百炼 / 通义千问、RAGAS、BM25（`rank_bm25`）、`jieba`。

> 本项目用于学习与工程实践。商用前请自行补全安全、合规与授权相关内容。
