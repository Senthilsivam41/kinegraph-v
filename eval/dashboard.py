"""
KineticGraph-Vectra — Live Telemetry Dashboard
Run with: streamlit run eval/dashboard.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KineticGraph-Vectra · Observability",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (allow dashboard to load even when packages not installed)
# ---------------------------------------------------------------------------
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False
    st.warning("plotly not installed — charts unavailable. `pip install plotly`")

# ---------------------------------------------------------------------------
# Custom CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f1117 0%, #1a1f2e 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Main background */
    .stApp { background: #0a0e1a; color: #e2e8f0; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px !important;
        backdrop-filter: blur(8px);
    }
    [data-testid="stMetricLabel"]  { color: #94a3b8 !important; font-size: 0.8rem; }
    [data-testid="stMetricValue"]  { color: #f1f5f9 !important; font-weight: 700; }
    [data-testid="stMetricDelta"]  { font-size: 0.75rem; }

    /* Section headers */
    h1 { color: #7c3aed !important; font-weight: 700; }
    h2 { color: #a78bfa !important; font-weight: 600; }
    h3 { color: #c4b5fd !important; font-weight: 500; }

    /* Dataframes */
    .dataframe { border-radius: 8px; overflow: hidden; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    time_window = st.slider("Time window (hours)", 1, 168, 24, 1)
    slow_threshold = st.slider("Slow query threshold (ms)", 500, 10000, 2000, 100)
    conf_threshold = st.slider("Low-confidence threshold", 0.1, 1.0, 0.5, 0.05)
    auto_refresh = st.checkbox("Auto-refresh (30 s)", value=False)
    if st.button("🔄 Refresh now"):
        st.rerun()
    st.markdown("---")
    st.markdown("### 📡 Data source")
    db_path = st.text_input("SQLite DB path", value="eval/metrics.db")
    st.markdown("---")
    st.caption("KineticGraph-Vectra · RAG Observability")

if auto_refresh:
    time.sleep(30)
    st.rerun()

# ---------------------------------------------------------------------------
# Load metrics from DB
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def _load_stats(db_path: str, hours: int) -> dict:
    try:
        from eval.metrics_collector import MetricsCollector
        mc = MetricsCollector(sqlite_path=db_path)
        return mc.get_dashboard_stats(time_window_hours=hours)
    except Exception as exc:
        return {"_error": str(exc)}


@st.cache_data(ttl=30)
def _load_slow(db_path: str, threshold: float) -> list:
    try:
        from eval.metrics_collector import MetricsCollector
        mc = MetricsCollector(sqlite_path=db_path)
        return mc.get_slow_queries(threshold_ms=threshold, limit=30)
    except Exception:
        return []


@st.cache_data(ttl=30)
def _load_low_conf(db_path: str, threshold: float) -> list:
    try:
        from eval.metrics_collector import MetricsCollector
        mc = MetricsCollector(sqlite_path=db_path)
        return mc.get_low_confidence_queries(threshold=threshold, limit=30)
    except Exception:
        return []


@st.cache_data(ttl=30)
def _load_mode_perf(db_path: str) -> dict:
    try:
        from eval.metrics_collector import MetricsCollector
        mc = MetricsCollector(sqlite_path=db_path)
        return mc.get_mode_performance()
    except Exception:
        return {}


stats = _load_stats(db_path, time_window)
slow_queries = _load_slow(db_path, slow_threshold)
low_conf = _load_low_conf(db_path, conf_threshold)
mode_perf = _load_mode_perf(db_path)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center'>🔭 KineticGraph-Vectra · RAG Observability</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center;color:#94a3b8'>Last {time_window}h window · "
    f"Auto-refresh: {'✅' if auto_refresh else '❌'}</p>",
    unsafe_allow_html=True,
)
st.divider()

if "_error" in stats:
    st.error(f"❌ Could not load metrics: {stats['_error']}")
    st.info("Make sure `eval/metrics.db` exists. Run some queries first to populate it.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
kpis = st.columns(5)
with kpis[0]:
    st.metric("Total Queries", stats.get("total_queries", 0))
with kpis[1]:
    lat = stats.get("latency_stats", {})
    st.metric("Avg Latency", f"{lat.get('avg_ms', 0):.0f} ms",
              delta=f"min {lat.get('min_ms', 0):.0f} / max {lat.get('max_ms', 0):.0f}")
with kpis[2]:
    hit = stats.get("context_hit_rate", 0)
    st.metric("Context Hit Rate", f"{hit*100:.1f}%",
              delta="↑ higher is better", delta_color="normal")
with kpis[3]:
    conf = stats.get("confidence_stats", {})
    st.metric("Avg Confidence", f"{conf.get('avg', 0):.3f}",
              delta=f"min {conf.get('min', 0):.2f} / max {conf.get('max', 0):.2f}")
with kpis[4]:
    tok = stats.get("token_stats", {})
    st.metric("Avg Tokens/Query", f"{tok.get('avg_per_query', 0):.0f}",
              delta=f"Total: {tok.get('total_in_window', 0):,}")

st.divider()

# ---------------------------------------------------------------------------
# Row 1 — Mode distribution + Agent latency
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧭 Query Mode Distribution")
    mode_dist = stats.get("mode_distribution", {})
    if mode_dist and _PLOTLY:
        fig = px.pie(
            names=list(mode_dist.keys()),
            values=list(mode_dist.values()),
            color_discrete_sequence=["#7c3aed", "#4f46e5", "#06b6d4"],
            hole=0.55,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.json(mode_dist)

with col2:
    st.markdown("### ⚡ Retrieval Agent Latency")
    agent_lat = stats.get("retrieval_agent_latency", {})
    if agent_lat and _PLOTLY:
        df_agents = pd.DataFrame([
            {"Agent": k, "Avg Latency (ms)": v["avg_ms"], "Calls": v["count"]}
            for k, v in agent_lat.items()
        ])
        fig2 = px.bar(
            df_agents, x="Agent", y="Avg Latency (ms)",
            color="Agent",
            color_discrete_sequence=["#7c3aed", "#4f46e5", "#06b6d4"],
            text="Avg Latency (ms)",
        )
        fig2.update_traces(texttemplate="%{text:.1f} ms", textposition="outside")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.json(agent_lat)

# ---------------------------------------------------------------------------
# Row 2 — Mode performance comparison
# ---------------------------------------------------------------------------
st.markdown("### 📊 Mode Performance Comparison")
if mode_perf and _PLOTLY:
    df_mode = pd.DataFrame([
        {
            "Mode": k,
            "Avg Latency (ms)": v["avg_latency_ms"],
            "Avg Chunks": v["avg_chunks"],
            "Queries": v["count"],
        }
        for k, v in mode_perf.items()
    ])
    fig3 = px.scatter(
        df_mode,
        x="Avg Latency (ms)",
        y="Avg Chunks",
        size="Queries",
        color="Mode",
        text="Mode",
        color_discrete_sequence=["#7c3aed", "#4f46e5", "#06b6d4"],
        size_max=60,
    )
    fig3.update_traces(textposition="top center")
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        margin=dict(t=20, b=40, l=20, r=20),
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.json(mode_perf)

# ---------------------------------------------------------------------------
# Row 3 — Slow queries
# ---------------------------------------------------------------------------
st.divider()
st.markdown(f"### 🐢 Slow Queries  ( > {slow_threshold} ms )")
if slow_queries:
    df_slow = pd.DataFrame(slow_queries)
    if _PLOTLY:
        fig4 = px.bar(
            df_slow.head(15),
            x="latency_ms",
            y="query",
            orientation="h",
            color="latency_ms",
            color_continuous_scale=["#4f46e5", "#ef4444"],
            labels={"latency_ms": "Latency (ms)", "query": "Query"},
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(
        df_slow[["query", "mode", "latency_ms", "chunk_count", "created_at"]],
        use_container_width=True,
    )
else:
    st.success(f"✅ No queries exceeded {slow_threshold} ms in the last {time_window}h")

# ---------------------------------------------------------------------------
# Row 4 — Low confidence answers
# ---------------------------------------------------------------------------
st.divider()
st.markdown(f"### ⚠️ Low-Confidence Answers  ( < {conf_threshold} )")
if low_conf:
    df_lc = pd.DataFrame(low_conf)
    if _PLOTLY:
        fig5 = px.histogram(
            df_lc, x="confidence", nbins=20,
            color_discrete_sequence=["#f59e0b"],
        )
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(t=10, b=20, l=20, r=20),
        )
        st.plotly_chart(fig5, use_container_width=True)
    st.dataframe(
        df_lc[["query", "mode", "confidence", "tokens_used", "created_at"]],
        use_container_width=True,
    )
else:
    st.success(f"✅ No low-confidence answers below {conf_threshold} in this window")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.78rem'>"
    "KineticGraph-Vectra · RAG Observability Dashboard · "
    "Built with Streamlit + Plotly · "
    "Data from eval/metrics.db</p>",
    unsafe_allow_html=True,
)
