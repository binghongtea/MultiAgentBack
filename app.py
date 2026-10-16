"""启动：python -m streamlit run app.py"""

from dataclasses import asdict
from pathlib import Path

import streamlit as st

from rag.engine import KnowledgeBase
from rag.llm import ModelConfig

ROOT = Path(__file__).resolve().parent


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"查看引用原文（{len(sources)} 条）"):
        for i, source in enumerate(sources, 1):
            page = f" · 第 {source['page']} 页" if source["page"] else ""
            st.caption(
                f"[{i}] {source['source']}{page} · 片段 {source['chunk_number']}"
                f" · 相似度 {source['score']:.3f}"
            )
            st.text(source["text"])


def import_files(kb: KnowledgeBase, files: list[tuple[str, bytes]]) -> None:
    notices = []
    for name, content in files:
        try:
            action, count = kb.add_document(name, content)
            notices.append(("success", f"{name}：{action}，共 {count} 个片段。"))
        except ValueError as exc:
            notices.append(("error", f"{name}：{exc}"))
    st.session_state.notices = notices


def main() -> None:
    st.set_page_config(page_title="RAG 知识库", page_icon="📚", layout="wide")
    kb = KnowledgeBase()
    config = ModelConfig.from_env()
    st.session_state.setdefault("messages", [])
    documents = kb.list_documents()

    with st.sidebar:
        st.title("📚 我的知识库")
        st.caption("本地检索 · DeepSeek 问答")
        with st.form("upload_form", clear_on_submit=True):
            uploads = st.file_uploader(
                "上传知识文档", type=["txt", "md", "pdf"], accept_multiple_files=True,
                help="每个文件最多 20 MB。同名文件会更新，重复内容不会重复导入。",
            )
            submitted = st.form_submit_button("导入知识库", use_container_width=True)
        if submitted:
            if uploads:
                with st.spinner("正在解析并保存文档……"):
                    import_files(kb, [(file.name, file.getvalue()) for file in uploads])
                st.rerun()
            else:
                st.warning("请先选择文件。")

        if st.button("导入示例文档", use_container_width=True):
            example = ROOT / "examples" / "员工手册.md"
            import_files(kb, [(example.name, example.read_bytes())])
            st.rerun()

        for level, notice in st.session_state.pop("notices", []):
            getattr(st, level)(notice)

        if documents:
            with st.expander(f"管理文档（{len(documents)}）"):
                by_id = {doc["id"]: doc for doc in documents}
                selected = st.selectbox(
                    "选择文档", options=list(by_id),
                    format_func=lambda doc_id: (
                        f"{by_id[doc_id]['name']}（{by_id[doc_id]['chunk_count']} 个片段）"
                    ),
                )
                if st.button("删除选中文档", use_container_width=True):
                    kb.delete_document(selected)
                    st.session_state.notices = [("success", "文档已删除。")]
                    st.rerun()

        with st.expander("检索设置"):
            top_k = st.slider("最多引用片段数", 1, 8, 3)
            min_score = st.slider("最低相似度", 0.0, 0.8, 0.05, 0.01,
                                  help="数值越大越严格；没有检索结果时可适当降低。")
        st.caption(f"当前模型：{config.model_name or '未设置'}")
        st.caption("已配置 API Key" if config.api_key else "尚未配置 API Key")
        st.caption("配置文件：项目根目录 .env")
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("RAG 知识库问答")
    st.write("上传文档，围绕知识库提问，并查看回答引用的原文。")
    left, middle, right = st.columns(3)
    left.metric("知识文档", len(documents))
    middle.metric("文本片段", sum(doc["chunk_count"] for doc in documents))
    right.metric("检索方式", "TF-IDF")

    if not config.api_key or not config.model_name:
        st.info("先在根目录 .env 填写 API_KEY 和 MODEL_NAME。配置前也可以导入文档并查看检索原文。")
    if not documents:
        st.info("从左侧上传 TXT、Markdown 或 PDF，也可以点击「导入示例文档」开始体验。")
    elif not st.session_state.messages:
        st.caption("示例问题：年假有多少天？｜报销需要哪些材料？｜远程办公怎么申请？")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("warning"):
                st.warning(message["warning"])
            st.markdown(message["content"])
            show_sources(message.get("sources", []))

    if question := st.chat_input("输入关于知识库的问题……", disabled=not documents, max_chars=2000):
        if not question.strip():
            st.warning("请输入有效的问题。")
            return
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("正在检索并生成回答……"):
                answer = kb.ask(question, config, top_k=top_k, min_score=min_score)
            if answer.warning:
                st.warning(answer.warning)
            st.markdown(answer.text)
            sources = [asdict(hit) for hit in answer.sources]
            show_sources(sources)
        st.session_state.messages.append({
            "role": "assistant", "content": answer.text,
            "sources": sources, "warning": answer.warning,
        })


if __name__ == "__main__":
    main()
