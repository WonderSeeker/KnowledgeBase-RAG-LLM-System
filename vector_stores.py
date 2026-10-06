import logging

from langchain_chroma import Chroma
from chromadb.config import Settings
import config_data as config

# 1) 关掉匿名遥测的数据发送
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)


def _silence_chroma_telemetry_noise() -> None:
    """消掉 chromadb 0.4.24 的遥测报错噪声（不是我们触发的异常）

    根因：chromadb 0.4.24 的 posthog 适配层用的是老 API
        posthog.capture(user_id, event_name, properties)      # 3 个位置参数
    而新版 posthog 的 capture 只接受 1 个位置参数，于是每次上报都在签名校验阶段抛
        TypeError: capture() takes 1 positional argument but 3 were given
    并被 logger.error 打出来。注意 Settings(anonymized_telemetry=False) 只能阻止"真正发送"
    （它只是把 posthog.disabled 置 True），挡不住这行日志——异常发生在真正发送之前。
    所以这里直接把该 logger 提到 CRITICAL：既不发送，也不再刷日志。
    """
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


_silence_chroma_telemetry_noise()

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
