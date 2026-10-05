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

## 三、阶段 2/3/4

（未开始，待阶段 1 验收后填写）

---

## 四、禁止事项（改造红线，勿破）

1. 不重建 `.venv`，不换成 Python 3.12（`chroma-hnswlib==0.7.3` 在 3.12 上只能现场编译 C++，会失败）。
2. 不改 `requirements.txt` 的版本锁定。
3. `requirements.txt` 含中文注释，直接 pip 装会 `UnicodeDecodeError: 'gbk'`，需先 `$env:PYTHONUTF8='1'`。
4. `DASHSCOPE_API_KEY` 只放 `.env`，禁止写进代码或提交。
5. 不下载任何 GitHub Release 文件（通道被阻断）。
