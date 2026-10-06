
"""全局配置：模型 / 向量库 / 分块 / 检索参数

改造说明（阶段 1）：
1. 新增 .env 加载。原项目装了 python-dotenv 却从未调用 load_dotenv()，
   所以 .env 里的 DASHSCOPE_API_KEY 根本读不到，只能依赖系统环境变量。
2. 路径改为基于本文件位置的绝对路径。原来是 "./xxx" 相对当前工作目录，
   从不同目录启动会读写到不同的库（README 里的 Q1 就是这个坑）。
"""
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# 加载项目根目录下的 .env（只存密钥，已在 .gitignore 中排除）
load_dotenv(BASE_DIR / ".env")

md5_path = str(BASE_DIR / "md5.text")

# Chroma
collection_name="rag"
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

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

#
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}