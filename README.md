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

### 效果结论（真实嵌入 512 维 + 确定性 IR 指标，29 条文档级标注）

指标口径：**文档级** —— 每个问题的正例 = 评测集中标注的应命中文档（人工出处，非循环）。
样本数 n=29。完整数据与局限见 `CHANGES.md` §4.3-R / §4.4-R。

**检索策略（统一 k=4，语料 6 片段）**

| 指标 | vector | bm25 | hybrid |
|---|---|---|---|
| doc_hit_rate@1 | 1.0000 | **0.8966** | 1.0000 |
| doc_mrr | 1.0000 | 0.9138 | 1.0000 |

BM25 在三个不同设置（1000/100 + k=4、300/30 + k=4、150/15 + k=1）下都稳定落后
（分别差 **3 题、1 题、1 题**，即 26/29 与 28/29），方向一致；
落后的原因可定位到语义改写类问题（如「怎样穿显得更瘦？」向量命中、BM25 零命中）。

**分块参数（固定 hybrid + k=4）**

| 指标 | 150/15（32 片段） | 300/30（15） | 600/60（9） | 1000/100（6） |
|---|---|---|---|---|
| chunk_precision@3 | **0.9080** | 0.7701 | 0.7011 | 0.6092 |

分块越细，top-k 里的噪声越少（单调、可复现）。

**必须一起读的局限**（否则这些数字会被过度解读）：

- **n=29，差 1 题 = 3.4%，不构成统计显著**。可复现的是"方向一致"，不是"显著更优"。
- **文档级指标只有 3 份文档，接近饱和**（`doc_hit@1` 普遍 1.0），区分度有限。
- **`recall@k` 不可跨分块配置比较**：它随分块变大而"变好"是语料变小所致，不是检索变好。
  这也是报告同时输出 `precision@k` 的原因。
- 片段级（`chunk_*`）是**弱标注**口径，正例由嵌入相似度定义，**结构性偏袒嵌入型检索器**，
  不能当作公平对比的证据。

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
├─ file_history_store.py      会话历史文件存储
├─ config_data.py             全局配置
├─ scripts/
│  └─ ingest.py               命令行入库（支持分块参数 / 集合隔离 / --dry-run）
├─ evaluation/
│  ├─ dataset.py              评测集构建（29 条内置样本 + JSONL 读写）
│  ├─ eval_set.jsonl          权威评测集（可直接手工增删）
│  ├─ run_ragas.py            跑评测：链路产出 + RAGAS 打分 + 落 JSON
│  ├─ offline.py              离线替身（本地切分 / 字面检索 / 启发式代理指标）
│  ├─ report.py               报告与对比表（含指标口径说明小节）
│  ├─ experiments.py          对比实验驱动（分块参数 × 检索策略）
│  ├─ requirements-eval.txt   评测依赖（ragas 0.1.21）
│  └─ constraints-eval.txt    主依赖版本约束（防误升级）
├─ requirements.txt           原项目依赖（版本锁定未改动）
├─ requirements-optim.txt     检索优化依赖（rank-bm25 + jieba）
├─ assets/                    示例语料与 README 配图
├─ CHANGES.md                 改造记录：原项目 / 改了什么 / 量化结果 / 结论有效性划定
└─ LICENSE
```

---

## 许可与致谢

- 本项目基于 [lhh737/KnowledgeBase-RAG-LLM-System](https://github.com/lhh737/KnowledgeBase-RAG-LLM-System)（MIT）改造，遵循 MIT 许可。
- 致谢：Streamlit、LangChain、Chroma、阿里云百炼 / 通义千问、RAGAS、BM25（`rank_bm25`）、`jieba`。

> 本项目用于学习与工程实践。商用前请自行补全安全、合规与授权相关内容。
