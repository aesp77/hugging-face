"""Page 3: FinBERT sentiment analysis on financial headlines."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(_project_root) / "src"))
sys.path.insert(0, str(Path(_project_root) / "streamlit_app"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from hf_volsurf.data.loaders import VolSurfaceDataLoader
from hf_volsurf.models.finbert import FinBERTSentiment
from hf_volsurf.utils.vol_math import STRIKE_GRID
from utils.config import DB_PATH

st.set_page_config(page_title="FinBERT Sentiment", layout="wide")
st.title("FinBERT Sentiment & Volatility")
st.markdown("Score financial headlines with FinBERT and see the relationship with implied volatility.")


@st.cache_resource
def load_models():
    sentiment = FinBERTSentiment()
    loader = VolSurfaceDataLoader(DB_PATH)
    grids, dates = loader.get_all_surface_grids()
    atm_idx = STRIKE_GRID.index(1.0)
    atm_1m = grids[:, 0, atm_idx]
    return sentiment, loader, dates, atm_1m


sentiment_model, loader, all_dates, atm_1m_series = load_models()

# --- Headline input ---
st.subheader("Analyse a Headline")
headline = st.text_input(
    "Financial headline",
    value="Federal Reserve raises interest rates by 75 basis points",
)

if st.button("Analyse", type="primary"):
    results = sentiment_model.score([headline])
    r = results[0]

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = go.Figure(data=[go.Bar(
            x=["positive", "negative", "neutral"],
            y=[r.positive, r.negative, r.neutral],
            marker_color=["green", "red", "gray"],
        )])
        fig.update_layout(
            title=f"FinBERT Sentiment (net: {r.net_sentiment:+.3f})",
            yaxis_title="Score",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Positive", f"{r.positive:.4f}")
        st.metric("Negative", f"{r.negative:.4f}")
        st.metric("Neutral", f"{r.neutral:.4f}")
        st.metric("Net Sentiment", f"{r.net_sentiment:+.4f}")

# --- Batch analysis ---
st.divider()
st.subheader("Batch Analysis — Sample Headlines")

sample_headlines = [
    "WHO warns coronavirus outbreak approaching pandemic levels",
    "Pfizer vaccine shows 90% efficacy sending stocks soaring",
    "Silicon Valley Bank collapses in largest bank failure since 2008",
    "Nvidia surpasses $2 trillion market cap on AI boom",
    "Global markets plunge on Japan carry trade unwind fears",
    "Fed cuts rates by 50 basis points in first reduction since 2020",
    "S&P 500 hits all-time high on strong economic data",
    "Market volatility spikes amid banking sector concerns",
]

if st.button("Run Batch Analysis"):
    with st.spinner("Scoring 8 headlines..."):
        df = sentiment_model.score_df(sample_headlines)
    st.dataframe(
        df.style.background_gradient(subset=["net_sentiment"], cmap="RdYlGn"),
        use_container_width=True,
    )

# --- ATM vol context ---
st.divider()
st.subheader("ATM 1m Vol Context")
st.markdown("For reference — the ATM 1m implied volatility time series (2010-2026).")

import matplotlib.pyplot as plt

fig_ts, ax = plt.subplots(figsize=(12, 3))
ax.plot(range(len(all_dates)), atm_1m_series, linewidth=0.5, color="steelblue")
ax.set_ylabel("ATM 1m IV")
tick_positions = list(range(0, len(all_dates), len(all_dates) // 8))
ax.set_xticks(tick_positions)
ax.set_xticklabels([all_dates[i][:7] for i in tick_positions], rotation=45, fontsize=8)
ax.grid(True, alpha=0.3)
st.pyplot(fig_ts)
