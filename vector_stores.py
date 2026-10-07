from langchain_chroma import Chroma
import config_data as config
import embeddings as embeddings_module

# 客户端设置与遥测治理已抽到 chroma_quiet.py（入口应显式 import 它），这里 re-export 保持兼容：
#   from vector_stores import CHROMA_SETTINGS   —— 这行在中英多个文件里都在用
from chroma_quiet import CHROMA_SETTINGS        # noqa: F401

class VectorStoreService(object):
    def __init__(self,embedding, collection_name=None):
        """
        :param embedding: 嵌入模型的传入
        :param collection_name: 集合名（阶段 3 做分块对比实验时，不同分块参数各用一个集合）
        """
        self.embedding= embedding
        # 默认集合名也按嵌入后端派生（rag / rag_local），避免"检索侧用了 dashscope 的集合"这类错配
        self.collection_name = collection_name or embeddings_module.default_collection_name()

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
            client_settings=CHROMA_SETTINGS,
        )

    def get_retriever(self, k=None):
            """返回向量检索器，方便加入chain

            :param k: 本次检索返回的片段数，默认取 config.retrieve_k
                      （混合检索/重排需要先粗排一批候选，会传入更大的值）
            """
            return self.vector_store.as_retriever(
                search_kwargs={"k": config.retrieve_k if k is None else k}
            )

if __name__ =='__main__':
        from langchain_community.embeddings import DashScopeEmbeddings
        retriever= VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

        res= retriever.invoke("我的身高180，尺码推荐")
        print(res)
