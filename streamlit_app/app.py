"""HF VolSurf: Volatility Surface Explorer — Streamlit App.

Main entry point. Run with:
    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

# sys.path setup per project-scaffold skill
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="HF VolSurf Explorer",
    page_icon="📈",
    layout="wide",
)

st.title("HF VolSurf: Volatility Surface Explorer")
st.markdown(
    """
    Explore **16 years of SPX implied volatility surfaces** using models
    from the Hugging Face ecosystem.

    **Pages:**
    | Page | What it does |
    |------|-------------|
    | **Surface Explorer** | Browse real vol surfaces by date, 3D interactive plots |
    | **DDPM Generator** | Generate new surfaces from noise, check arbitrage |
    | **FinBERT Sentiment** | Score financial headlines, correlate with vol |
    | **Data Summary** | Database overview, coverage, split info |

    ---

    **Data:** 429,936 vol surface grid points (SPX, 2010-2026, Marquee).
    Copied from `rl_hedging_comparison`.

    **Models:**
    - **FinBERT** — ProsusAI/finbert (sentiment, frozen)
    - **DDPM** — UNet2D 651K params (generative surfaces)
    """
)
