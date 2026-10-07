import logging

from langchain_chroma import Chroma
from chromadb.config import Settings
import config_data as config
import embeddings as embeddings_module

# 1) 关掉匿名遥测的数据发送
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    # ★★ 必须显式打开持久化，否则是"进程内内存库"（数据不落盘，且进程内自测永远通过！）
    #
    # 踩坑记录：langchain_chroma.Chroma 只有在**不传** client_settings 时才会自己构造
    #     Settings(is_persistent=True, persist_directory=persist_directory)
    # 一旦传入 client_settings，它就照用不误、不会补 is_persistent。
    # 而 chromadb 的 Settings 默认 is_persistent=False → 退化成本进程内存库：
    #     - 入库进程内 count() 正常（自己读自己写的）
    #     - 换个进程就查不到集合，数据全丢
    # 症状极隐蔽：进程内自测 100% 通过，只有跨进程（入库一个进程、检索另一个进程）才暴露。
    is_persistent=True,
    persist_directory=config.persist_directory,
)


def _silence_chroma_telemetry_noise() -> None:
    """消掉 chromadb 0.4.24 的遥测报错噪声（不是我们触发的异常）"""
    # 故意把该 logger 提到 CRITICAL：chromadb 0.4.24 的 posthog 适配层
    # 用 3 个位置参数调 posthog.capture，新版 posthog 只接受 1 个，
    # 异常发生在签名校验阶段，anonymized_telemetry=False 挡不住这行 logger.error。
    # ★ 不要"修好"它——我们完全不需要遥测。
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


_silence_chroma_telemetry_noise()

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
