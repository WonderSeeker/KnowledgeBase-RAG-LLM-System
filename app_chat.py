import streamlit as  st
from rag import RagService
import config_data as config
import time

# 标题
st.title("智能客服")
# 分隔符
st.divider()

# 避免性能压力，session_state 存入对象
if "message" not in st.session_state:
    st.session_state["message"]=[{"role":"assistant","content":"你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"]= RagService()
#循环 输出历史信息，原本只记录但页面不显示
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
# 在页面最下方提供用户输入栏
prompt= st.chat_input()

if prompt :
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list= []
    start_time= time.perf_counter()      # 基线指标：记录端到端耗时
    with st.spinner("AI 思考中......."):
        # 调用 RAG服务

        # 直接输出
        # res= st.session_state["rag"].chain.invoke({"input":prompt},config.session_config)
        #
        # st.chat_message("assistant").write(res)
        # st.session_state["message"].append({"role":"assistant","content":res})

        # 流式输出
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})

    # 基线观测：页面上直接显示耗时，同时打进日志（D:\Projects\_logs\rag.out）
    cost= time.perf_counter() - start_time
    st.caption(f"本次回答耗时 {cost:.2f}s")
    print(f"[问答] 端到端耗时 {cost:.2f}s")

