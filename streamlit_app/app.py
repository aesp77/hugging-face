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

st.markdown("""
### What is this?

This app explores **implied volatility surfaces** — the 3D map of how the market prices
uncertainty across different option strikes and maturities. Understanding vol surfaces is
central to derivatives pricing, risk management, and volatility trading.

We use **Hugging Face** models to analyse, predict, and generate these surfaces, demonstrating
how the HF ecosystem (Transformers, Diffusers, Datasets, Gradio) applies to quantitative finance.

### What did we learn?

| Finding | Notebook | Implication |
|---------|----------|-------------|
| Vol surfaces are **highly persistent** day-to-day | NB 04 | Naive "yesterday = today" is a strong baseline; Transformers need to predict *changes*, not levels |
| **FinBERT sentiment correlates with vol** (r = -0.575) | NB 03 | Negative headlines align with vol spikes — sentiment is a useful conditioning feature |
| **DDPM generates realistic surfaces** (80% arb-free) | NB 05 | The model learns no-arbitrage structure from data alone, matching real data's 78% rate |
| **VIX is highly correlated with ATM vol** (r = 0.81) | NB 02 | VIX captures the "level" of the surface — useful for conditioning generative models |

### Pages

| Page | HF Library | Purpose |
|------|-----------|---------|
| **Surface Explorer** | — | Browse 4,134 real SPX surfaces interactively |
| **DDPM Generator** | `diffusers` | Generate new surfaces from noise, validate with arb checks |
| **FinBERT Sentiment** | `transformers` | Score financial headlines and relate to vol dynamics |
| **Data Summary** | — | Database overview, train/val/test split, quality metrics |

---

**Data:** 429,936 vol surface grid points (SPX, 2010-2026, source: Marquee via `rl_hedging_comparison`).
Each surface is an 8-tenor x 13-strike grid (1m-2y maturity, 70%-130% moneyness).
""")

