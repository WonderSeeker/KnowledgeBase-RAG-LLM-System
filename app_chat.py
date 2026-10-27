import streamlit as  st
from rag import RagService
from knowledge_base import KnowledgeBaseService, extract_text
import config_data as config
import time

# 标题
st.title("智能客服")
# 分隔符
st.divider()

# ===================== 侧边栏：知识库更新 =====================
# 改造动机：原来上传在 app_upload.py（另一个 Streamlit 进程），写进去的向量
#   对已运行的聊天进程不可见（Chroma 索引驻留进程内存），必须重启聊天页才能检索到。
#   放到同一进程后：本页上传 → 立即可检索，无需重启。
#   已实测：同进程内新增的片段，对已有 collection 句柄与新建 client 都可见。
with st.sidebar:
    st.header("知识库更新")
    st.caption("支持 TXT / PDF / DOCX，可多选。上传在本页进程内完成，无需重启即可检索。")

    # 用 form 把「选文件」和「开始入库」绑在一起，避免 Streamlit 每次交互重跑时
    # 重复处理同一个文件（重复上传虽然会被 md5 挡住，但会白跑一次解析）。
    with st.form("kb_upload_form", clear_on_submit=True):
        files = st.file_uploader(
            "选择要入库的文件",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("开始入库")

    if submitted:
        if not files:
            st.warning("还没有选择文件")
        else:
            if "kb" not in st.session_state:
                st.session_state["kb"] = KnowledgeBaseService()   # 只初始化一次（内含一次嵌入维度探测）
            for f in files:
                try:
                    text = extract_text(f.getvalue(), f.name)
                except Exception as exc:
                    st.error(f"{f.name} 解析失败：{exc}")
                    continue
                if not text or not text.strip():
                    st.error(f"{f.name}：没抽到文字（可能是扫描件/图片版，需先 OCR）")
                    continue
                with st.spinner(f"入库 {f.name} …"):
                    result = st.session_state["kb"].upload_by_str(text, f.name)
                if result.startswith("[Success]"):
                    st.success(f"{f.name}：{result}")
                else:
                    st.info(f"{f.name}：{result}")

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

