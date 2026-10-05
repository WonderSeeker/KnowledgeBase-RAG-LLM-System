
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

max_spliter_char_number= 1000  # 文本分割阈值

# 检索返回的片段数量（k）
retrieve_k = 1     # ⚠️ 基线值 1：只把 1 个片段喂给大模型，是 context_recall 的天花板
similarity_threshold = retrieve_k   # 兼容旧名（原项目拿这个名字当 k 用，命名与用途不符）

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

#
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}