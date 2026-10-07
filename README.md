# KnowledgeBase RAG LLM System

在本地上传文档 → 切分入库 → 检索增强问答，并**带一套评测层把"效果好坏"量化出来**的工程化 RAG 项目。

---

## 项目状态

**已完成并验证**（本仓库可复现的证据）：

- 检索层：向量 / BM25 / RRF 混合 / 重排，统一 `invoke()` 契约 —— 8 项探针通过（CHANGES §6.1）
- 修复 3 个真实缺陷（含一个使问答链路完全不可用的 ImportError）（CHANGES §2.1 / §4.8 / §5.2）
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
| `models` | `embedding` / `chat` 模型名 |
| `version` | 接口版本 |

### `POST /chat`

请求：`{"question": "...", "session_id": "..."}`（`question` 必填，长度 1~2000；`session_id` 用于多轮历史，默认 `api_user`）

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

另有一个配置层缺陷：`.env` 从未被加载（依赖里装了 `python-dotenv` 却没有任何一处 `load_dotenv()`），
导致按文档配置 `.env` 的用户拿不到 Key —— 见 `CHANGES.md` §2.1。

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

> 检索策略与分块参数的效果对比：待接入真实嵌入后补充（复测口径见 `CHANGES.md` §6.2）。

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
