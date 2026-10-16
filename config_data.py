
"""全局配置：模型 / 向量库 / 分块 / 检索参数

改造说明（阶段 1）：
1. 新增 .env 加载。原项目装了 python-dotenv 却从未调用 load_dotenv()，
   所以 .env 里的 DASHSCOPE_API_KEY 根本读不到，只能依赖系统环境变量。
2. 路径改为基于本文件位置的绝对路径。原来是 "./xxx" 相对当前工作目录，
   从不同目录启动会读写到不同的库（README 里的 Q1 就是这个坑）。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# 加载项目根目录下的 .env（只存密钥，已在 .gitignore 中排除）
load_dotenv(BASE_DIR / ".env")

md5_path = str(BASE_DIR / "md5.text")

# Chroma
collection_name="rag"          # 集合名基名；实际名字按嵌入后端派生（dashscope -> rag，local -> rag_local）
collection_override = None     # 显式指定集合名时优先（分块对比实验用），为 None 则自动派生
persist_directory=str(BASE_DIR / "chroma_db")

# spliter
chunk_size= 1000
chunk_overlap= 100
separators =["\n\n","\n",".","!","?","。","！","？"," ",""]

# ============================ DEPRECATED ============================
# 保留仅为兼容旧配置文件，逻辑已不再读取。
# 改这个值不会有任何效果，请不要再基于它做任何判断。
#
# 原语义：「文本分割硬上限」，≤ 该值的文档一律不切分。
# 为什么废弃：该门槛曾导致 chunk_size 对短文档完全失效——
#   《尺码推荐.txt》270 字，chunk_size 设 300 或 1500 都只有 1 个片段（详见 CHANGES.md 4.8）。
# 现在分块判定一律交给 RecursiveCharacterTextSplitter，chunk_size/chunk_overlap 真实生效。
# ====================================================================
max_spliter_char_number= 1000

# 检索返回的片段数量（k）
retrieve_k = 1     # 基线值 1：只把 1 个片段喂给大模型，是 context_recall 的天花板
similarity_threshold = retrieve_k   # 兼容旧名（原项目拿这个名字当 k 用，命名与用途不符）

# ---------------- 阶段 3：检索策略 ----------------
# vector 纯向量 / bm25 纯关键词 / hybrid 混合检索（RRF 融合）
retrieval_method = "vector"
candidate_k = 10              # 粗排候选数：混合检索与重排都从这批候选里再做精选
rrf_k = 60                    # RRF 融合常数（经验值 60：越大越强调"两路检索都靠前"）
rerank_enabled = False        # 是否开启重排
rerank_model = "gte-rerank-v2"   # DashScope 重排模型（用 dashscope SDK 的 TextReRank 调用）

# ---------------- 阶段 4 补充：嵌入后端 ----------------
# dashscope 云端 / local 本地 OpenAI 兼容服务
# 默认保持 dashscope，现有行为不变。
# ⚠️ 两个后端的向量维度不同，绝不可混用同一个 collection：
#    代码会把「后端 + 模型 + 维度」写进 collection metadata，不一致直接报错退出。
#    默认集合名也随之区分：dashscope -> rag，local -> rag_local
embedding_backend = os.environ.get("EMBEDDING_BACKEND", "dashscope")
local_embedding_base_url = "http://127.0.0.1:11434"   # 启动：cd D:\Projects ; .\svc.ps1 start llm
local_embedding_model = "bge-small-zh-v1.5"           # 实测 512 维、L2 归一化
local_embedding_api_key = "dummy-not-used"            # 本地服务不校验密钥，占位即可

embedding_model_name="text-embedding-v4"
# ★ 2026-10-16：默认值由 qwen3-max 改为 qwen-max。
#   实测当前 Key 下 `qwen3-max` 返回 404 InvalidParameter: Model not exist，
#   而 qwen-max / qwen-plus / qwen-long 可用 —— 默认值若不改，默认后端（dashscope）下
#   每次问答都会 502。可用环境变量 CHAT_MODEL_NAME 覆盖。
chat_model_name = os.environ.get("CHAT_MODEL_NAME", "qwen-max")

# ---------------- 生成模型后端 ----------------
# 为什么要它：RAGAS 的 faithfulness / answer_relevancy 需要**真实生成的答案**，
#   而裁判 LLM 与生成 LLM 可以是两个来源。默认保持 dashscope（通义千问），行为与改造前完全一致。
#   openai-compatible：任何 OpenAI 兼容端点 —— DeepSeek、本地 11434（零成本对比）都能用它。
# ★ 注意：DeepSeek 等厂商**没有 embeddings 接口**，嵌入必须另找来源（本项目用本地 bge）。
chat_backend = os.environ.get("CHAT_BACKEND", "dashscope")          # dashscope | openai-compatible
openai_compatible_base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "https://api.deepseek.com/v1")
openai_compatible_model = os.environ.get("OPENAI_COMPATIBLE_MODEL", "deepseek-chat")
openai_compatible_api_key_env = os.environ.get("OPENAI_COMPATIBLE_API_KEY_ENV", "DEEPSEEK_API_KEY")

#
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}