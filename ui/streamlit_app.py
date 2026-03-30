import streamlit as st
import requests
import pandas as pd
import json
import time

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(layout="wide")

# -----------------------------
# 🎨 GLOBAL STYLING
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.answer-box {
    background: linear-gradient(145deg, #020617, #020617);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🤖 Self Optimizing RAG System")
st.caption("Retrieval • Memory • Optimization • Observability")

# -----------------------------
# 🔥 SMART ANSWER RENDERER
# -----------------------------
def render_answer(text, query_type):

    st.markdown("### 💬 Answer")

    if query_type == "coding":

        if "def " in text:
            lang = "python"
        elif "function" in text:
            lang = "javascript"
        else:
            lang = "text"

        # basic formatting
        formatted = text.replace(":", ":\n").replace(" for ", "\nfor ")

        st.code(formatted, language=lang)

    else:
        st.markdown(text)

# -----------------------------
# INPUT
# -----------------------------
question = st.chat_input("Ask anything...")

if question:

    with st.chat_message("user"):
        st.markdown(question)

    response = requests.post(API_URL, json={"question": question}, stream=True)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        generated_text = ""
        observability_data = ""

        for chunk in response.iter_content(chunk_size=1024):

            if chunk:
                text = chunk.decode("utf-8")

                if "__OBSERVABILITY_START__" in text:
                    parts = text.split("__OBSERVABILITY_START__")
                    generated_text += parts[0]
                    observability_data += parts[1] if len(parts) > 1 else ""
                    continue

                if observability_data:
                    observability_data += text
                else:
                    generated_text += text
                    placeholder.markdown(generated_text)
                    time.sleep(0.01)

        placeholder.empty()

    # -----------------------------
    # OBSERVABILITY LOAD FIRST
    # -----------------------------
    obs = {}
    if observability_data:
        obs = json.loads(observability_data)

    query_type = obs.get("query_analysis", {}).get("type", "general")

    # -----------------------------
    # 🔥 RENDER ANSWER
    # -----------------------------
    render_answer(generated_text, query_type)

    # -----------------------------
    # 📊 OBSERVABILITY
    # -----------------------------
    if obs:

        st.markdown("## 📊 Observability")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("🧠 Mode", obs.get("mode"))
        col2.metric("🤖 Model", obs.get("model_used"))
        col3.metric("📌 Query", query_type)
        col4.metric("🧠 Memory", obs.get("memory_used", 0))

        # -----------------------------
        # ⚙️ CONTROL PLANE
        # -----------------------------
        st.markdown("### ⚙️ Control Plane")

        col1, col2 = st.columns([2,1])

        with col1:
            st.markdown("#### 🧩 Active Configuration")
            st.json(obs.get("optimizer_config", {}))

        with col2:
            st.markdown("#### 🎯 Confidence")

            conf = obs.get("confidence", None)

            if conf is not None:
                st.progress(conf)

                if conf > 0.75:
                    st.success(f"{conf * 100:.1f}% • High Confidence")
                elif conf > 0.4:
                    st.warning(f"{conf * 100:.1f}% • Medium Confidence")
                else:
                    st.error(f"{conf * 100:.1f}% • Low Confidence")
            else:
                st.info("Pending evaluation")

        # -----------------------------
        # 📚 RETRIEVAL
        # -----------------------------
        st.markdown("### 📚 Retrieval")

        retrieval = obs.get("retrieval", {})

        col1, col2 = st.columns(2)
        col1.metric("Total Docs", retrieval.get("total_docs", 0))
        col2.metric("Top K", retrieval.get("top_k", 0))

        # -----------------------------
        # 🧮 RERANKER
        # -----------------------------
        if "reranker_scores" in obs:

            st.markdown("### 🧮 Reranker Insights")

            scores = [score for _, score in obs["reranker_scores"]]

            if scores:
                min_score = min(scores)
                max_score = max(scores)

                def normalize(s):
                    if max_score == min_score:
                        return 0.5
                    return (s - min_score) / (max_score - min_score)

                normalized_data = [
                    (doc, score, normalize(score))
                    for doc, score in obs["reranker_scores"]
                ]

                df = pd.DataFrame(
                    normalized_data,
                    columns=["Document", "Raw Score", "Relevance (0-1)"]
                )

                df = df.sort_values(by="Relevance (0-1)", ascending=False)

                st.dataframe(df, width="stretch")

        # -----------------------------
        # 📌 CONTEXT
        # -----------------------------
        if "final_context" in obs:

            st.markdown("### 📌 Retrieved Context")

            for i, doc in enumerate(obs["final_context"]):
                with st.expander(f"📄 Chunk {i+1}", expanded=(i == 0)):
                    preview = doc[:800] + "..." if len(doc) > 800 else doc
                    st.markdown(preview)

        # -----------------------------
        # ⏱️ LATENCY
        # -----------------------------
        if "latency" in obs:

            st.markdown("### ⏱️ Latency Breakdown")

            latency_df = pd.DataFrame(
                list(obs["latency"].items()),
                columns=["Stage", "Seconds"]
            )

            latency_df = latency_df.sort_values(by="Seconds", ascending=False)

            st.dataframe(latency_df, width="stretch")
            st.bar_chart(latency_df.set_index("Stage"), width="stretch")