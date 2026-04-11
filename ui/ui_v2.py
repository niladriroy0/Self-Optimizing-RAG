import streamlit as st
import requests
import pandas as pd
import json
import time
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# 🌐 CONFIG & CONSTANTS
# -----------------------------
API_URL = "http://127.0.0.1:8000"
st.set_page_config(
    page_title="Self-Optimizing RAG Control Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 🎨 PREMIUM CSS STYLE (GLASSMORPHISM & DARK MODE)
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@300;400;600;800&display=swap');

    :root {
        --primary: #6366f1;
        --bg-dark: #0f172a;
        --glass: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 0rem;
        padding-bottom: 160px !important;
    }

    /* Streamlit internal container override */
    .st-emotion-cache-tn0cau {
        display: flex;
        gap: 1rem;
        width: 100%;
        max-width: 100%;
        height: auto;
        min-width: 1rem;
        flex-flow: column;
        flex: 1 1 0%;
        -webkit-box-align: stretch;
        align-items: stretch;
        -webkit-box-pack: start;
        justify-content: start;
        padding-top: 0rem;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    /* Glass Cards */
    .glass-card {
        background: var(--glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #818cf8;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    /* Chat Bubbles */
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 18px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 85%;
        word-wrap: break-word;
        line-height: 1.5;
        font-size: 15px;
    }
    .user-bubble {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        margin-left: auto;
        color: white;
        border-bottom-right-radius: 4px;
        text-align: right;
    }
    .assistant-bubble {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        margin-right: auto;
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
        text-align: left;
    }
    
    /* Ensure the chat area has space for the fixed input */
    .chat-container {
        width: 100%;
    }

    /* Fix input at bottom but align with content */
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 20px;

        /* 👇 KEY FIX */
        left: 50%;
        transform: translateX(-50%);
        width: 70%;   /* adjust: 60–75% based on your taste */

        max-width: 900px;

        background: rgba(15, 23, 42, 0.95);
        padding: 10px 16px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);

        z-index: 9999;
        backdrop-filter: blur(12px);
    }

    /* Sidebar Fixes */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid var(--glass-border);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 15px;
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(16px);
        z-index: 1000;
        margin: 60px auto 20px auto;
        padding: 5px 40px;
        display: flex;
        justify-content: center;
        width: fit-content;
        border: 1px solid var(--glass-border);
        border-radius: 50px;
        gap: 80px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background-color: transparent !important;
        border: none !important;
        font-weight: 700;
        font-size: 22px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #818cf8 !important;
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        color: #818cf8 !important;
        border-bottom: 3px solid #818cf8 !important;
    }

    /* Sticky Sidebar Footer */
    .sidebar-footer {
        bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🛠️ HELPER FUNCTIONS
# -----------------------------
def get_config():
    try:
        r = requests.get(f"{API_URL}/config")
        return r.json().get("config", {})
    except:
        return {}

def update_config(new_params):
    try:
        requests.post(f"{API_URL}/config/update", json=new_params)
        st.success("Configuration updated successfully!")
    except Exception as e:
        st.error(f"Failed to update config: {e}")

def get_costs():
    try:
        r = requests.get(f"{API_URL}/cost")
        return r.json()
    except:
        return {}

def get_experiments():
    try:
        r = requests.get(f"{API_URL}/experiments")
        return r.json()
    except:
        return []

# -----------------------------
# 🏠 SIDEBAR (PERSISTENT LOGS & STATUS)
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=100)
    st.markdown("# System Status")
    
    status_col1, status_col2 = st.columns(2)
    status_col1.write("● API")
    status_col2.write("🟢 Active")
    
    status_col1, status_col2 = st.columns(2)
    status_col1.write("● Vector Store")
    status_col2.write("🟢 Ready")
    
    st.divider()

    if st.button("🗑️ Clear Conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.last_obs = {}
        st.rerun()
    
    st.divider()
    
    st.markdown("### 📊 Live Observability")
    obs_placeholder = st.empty()
    
    if 'last_obs' in st.session_state:
        obs = st.session_state.last_obs
        st.caption(f"Last Query Analysis: {obs.get('query_analysis', {}).get('type', 'N/A')}")
        st.progress(obs.get('confidence', 0.0), text=f"Confidence: {obs.get('confidence', 0.0)*100:.1f}%")
        
        latency = obs.get('latency', {})
        if latency:
            fig = px.bar(
                x=list(latency.values()), 
                y=list(latency.keys()), 
                orientation='h',
                template="plotly_dark",
                height=200,
                color_discrete_sequence=['#6366f1']
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Seconds",
                yaxis_title="",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=180,
                showlegend=False
            )
            fig.update_traces(marker_color='#818cf8')
            st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})
    
    st.markdown("---")
    st.markdown(
        "<div class='sidebar-footer'>"
        "<p style='font-size:11px; color:#64748b; opacity:0.8;'>"
        "Powered by Self-Optimizing RAG Engine • v2.0 Premium Dashboard"
        "</p>"
        "</div>",
        unsafe_allow_html=True
    )

# -----------------------------
# 📑 MAIN NAVIGATION (TABS)
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat Center", 
    "📈 Operations", 
    "⚙️ Settings", 
    "🧪 Experiments"
])

# -----------------------------
# TAB 1: CHAT CENTER
# -----------------------------
with tab1:
    st.title("Brainwave Express")
    st.caption("Self-Optimizing Retrieval Augmented Generation")
    
    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat History Container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        role_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
        st.markdown(f'<div class="chat-bubble {role_class}">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input loop
    if question := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f'<div class="chat-bubble user-bubble">{question}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            response = requests.post(f"{API_URL}/query", json={"question": question}, stream=True)
            
            placeholder = st.empty()
            full_content = ""
            
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    full_content += chunk.decode("utf-8")
                    
                    if "__OBSERVABILITY_START__" in full_content:
                        # Split by the marker
                        parts = full_content.split("__OBSERVABILITY_START__")
                        # Display only the part before the marker in the chat
                        placeholder.markdown(parts[0])
                    else:
                        # Continue displaying the stream
                        placeholder.markdown(full_content)

            # Final processing after stream ends
            if "__OBSERVABILITY_START__" in full_content:
                parts = full_content.split("__OBSERVABILITY_START__")
                final_answer = parts[0]
                obs_json = parts[1]
                
                # Save the final answer to message history
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
                try:
                    # Clean up the JSON string (strip any whitespace)
                    obs_data = json.loads(obs_json.strip())
                    st.session_state.last_obs = obs_data
                    st.rerun()
                except Exception as e:
                    # If JSON parsing fails, still save the message
                    st.error(f"Failed to parse observability data: {e}")
            else:
                st.session_state.messages.append({"role": "assistant", "content": full_content})

# -----------------------------
# TAB 2: OPERATIONS & COSTS
# -----------------------------
with tab2:
    st.title("Operations Dashboard")
    costs = get_costs()
    
    if costs:
        sess = costs.get("session", {})
        totals = costs.get("totals", {})
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Session Tokens", sess.get("session_total_tokens", 0))
        col2.metric("Session Cost", f"${sess.get('session_cost_usd', 0):.4f}")
        col3.metric("Total Tokens", totals.get("total_tokens", totals.get("total_input_tokens", 0) + totals.get("total_output_tokens", 0)))
        col4.metric("Total Cost", f"${totals.get('total_cost_usd', 0):.4f}")
        
        st.divider()
        
        st.subheader("Model Usage Breakdown")
        by_model = sess.get("by_model", {})
        if by_model:
            df_model = pd.DataFrame.from_dict(by_model, orient='index').reset_index()
            df_model.columns = ['Model', 'Requests', 'Input', 'Output', 'Cost']
            
            col_a, col_b = st.columns(2)
            with col_a:
                fig_pie = px.pie(df_model, values='Cost', names='Model', title="Cost Distribution", template="plotly_dark")
                st.plotly_chart(fig_pie, width="stretch")
            with col_b:
                fig_bar = px.bar(df_model, x='Model', y='Requests', title="Request Volume", template="plotly_dark", color='Model')
                st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("No session data yet. Start chatting to see metrics!")

# -----------------------------
# TAB 3: SETTINGS
# -----------------------------
with tab3:
    st.title("System Configuration")
    config = get_config()
    
    if config:
        st.markdown("### 🧩 Retrieval Strategy")
        c1, c2 = st.columns(2)
        top_k = c1.slider("Top K Documents", 1, 20, value=config.get("top_k", 5))
        reranker_top_k = c2.slider("Reranker Top K", 1, 50, value=config.get("reranker_top_k", 20))
        
        st.markdown("### 🤖 Model Parameters")
        c3, c4 = st.columns(2)
        temp = c3.slider("Temperature", 0.0, 1.0, value=float(config.get("temperature", 0.2)))
        max_tokens = c4.number_input("Max Tokens", 128, 4096, value=config.get("max_tokens", 512))
        
        st.markdown("### ⚡ Cognitive Features")
        f1, f2, f3, f4 = st.columns(4)
        use_reranker = f1.toggle("Use Reranker", value=config.get("use_reranker", True))
        hybrid = f2.toggle("Hybrid Search", value=config.get("enable_hybrid", True))
        multi_hop = f3.toggle("Multi-Hop Reason", value=config.get("enable_multi_hop", True))
        fallback = f4.toggle("Parametric Fallback", value=config.get("enable_fallback", True))

        st.markdown("### 🖥️ Hardware & Fallback Tuning")
        st.caption("Optimized for CPU-only environments (no GPU). Adjust thresholds to balance speed vs. quality.")
        h1, h2, h3 = st.columns(3)
        low_resource = h1.toggle(
            "Low Resource Mode",
            value=config.get("low_resource_mode", True),
            help="Reduces Reranker candidate cap from 20 → 5 to save CPU cycles."
        )
        min_relevance = h2.slider(
            "Min Relevance Threshold",
            min_value=0.0, max_value=1.0,
            value=float(config.get("min_relevance_threshold", 0.15)),
            step=0.01,
            help="Reranker score below this → skip RAG LLM call & go straight to fallback."
        )
        max_distance = h3.slider(
            "Max Retrieval Distance",
            min_value=0.5, max_value=3.0,
            value=float(config.get("max_retrieval_distance", 1.5)),
            step=0.1,
            help="Chroma L2 distance above this → document is too far from query (filtered out)."
        )
        
        if st.button("Apply Changes", type="primary"):
            update_config({
                "top_k": top_k,
                "reranker_top_k": reranker_top_k,
                "temperature": temp,
                "max_tokens": max_tokens,
                "use_reranker": use_reranker,
                "enable_hybrid": hybrid,
                "enable_multi_hop": multi_hop,
                "enable_fallback": fallback,
                "low_resource_mode": low_resource,
                "min_relevance_threshold": min_relevance,
                "max_retrieval_distance": max_distance,
            })
    else:
        st.warning("Could not fetch system configuration.")

# -----------------------------
# TAB 4: EXPERIMENTS
# -----------------------------
with tab4:
    st.title("Performance Leaderboard")
    exps = get_experiments()
    
    if exps:
        df_exps = pd.DataFrame(exps)
        
        st.markdown("### 🏆 Top Configurations")
        # Calc average score
        df_exps['Composite Score'] = (df_exps['answer_relevance'] + df_exps['faithfulness']) / 2
        
        avg_scores = df_exps.groupby('config')['Composite Score'].mean().sort_values(ascending=False).reset_index()
        st.table(avg_scores.head(5))
        
        st.divider()
        
        st.markdown("### 🧪 Raw History")
        st.dataframe(df_exps, width="stretch")
        
        st.markdown("### 📉 Metric Trends")
        fig_trend = px.line(df_exps, x='timestamp', y=['answer_relevance', 'faithfulness'], title="Quality over Time", template="plotly_dark")
        st.plotly_chart(fig_trend, width="stretch")
        
    else:
        st.info("No experiment logs found in experiments.db yet.")

