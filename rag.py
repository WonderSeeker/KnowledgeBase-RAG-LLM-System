import time
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt


class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                 "简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()

    def retrieve(self, question: str) -> list[Document]:
        """单独执行一次检索，返回命中的文档片段

        从链里把检索步骤抽出来，供脚本 / 评测 / 调试直接调用（阶段 2 的 RAGAS 需要拿到 contexts）
        """
        return self.vector_service.get_retriever().invoke(question)

    def answer(self, question: str, session_id: str = "user_001") -> dict:
        """一次性问答，返回答案 + 命中片段 + 分段耗时（供命令行与评测使用）

        与 Streamlit 界面走的是同一套检索器和同一条链，保证指标可比
        """
        total_start = time.perf_counter()

        retrieve_start = time.perf_counter()
        contexts = self.retrieve(question)
        retrieve_cost = time.perf_counter() - retrieve_start

        generate_start = time.perf_counter()
        text = self.chain.invoke(
            {"input": question},
            {"configurable": {"session_id": session_id}},
        )
        generate_cost = time.perf_counter() - generate_start

        return {
            "question": question,
            "answer": text,
            "contexts": [doc.page_content for doc in contexts],
            "n_contexts": len(contexts),                     # 基线指标：检索命中片段数
            "retrieve_seconds": round(retrieve_cost, 2),
            "generate_seconds": round(generate_cost, 2),
            "total_seconds": round(time.perf_counter() - total_start, 2),
        }

    def __get_chain(self):
        """获取最终的执行链"""

        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            # 基线观测：每次检索实际命中的片段数（当前 = config.retrieve_k）
            print(f"[检索] 命中片段数={len(docs)}，来源={[d.metadata.get('source') for d in docs]}")
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value: dict)->str:

            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value


        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            }| RunnableLambda(format_for_prompt_template) |self.prompt_template | print_prompt |self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(       # 增强的链
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == '__main__':
    # session id 配置
    session_config ={
        "configurable":{
            "session_id":"user_001",
        }
    }
    res = RagService().chain.invoke({"input":"我之前问了什么"},session_config)
    print(res)

