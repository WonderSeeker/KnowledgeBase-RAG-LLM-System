from langchain_chroma import Chroma
from chromadb.config import Settings
import config_data as config

# 关掉 Chroma 的匿名遥测：它每次操作都会往 stderr 打一行 "Failed to send telemetry event ..."
# （新版 posthog 与 chromadb 0.4.x 不兼容），把服务日志刷得没法看
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

class VectorStoreService(object):
    def __init__(self,embedding, collection_name=None):
        """
        :param embedding: 嵌入模型的传入
        :param collection_name: 集合名（阶段 3 做分块对比实验时，不同分块参数各用一个集合）
        """
        self.embedding= embedding
        self.collection_name = collection_name or config.collection_name

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
