"""Page 1: Browse real SPX vol surfaces by date."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(_project_root) / "src"))
sys.path.insert(0, str(Path(_project_root) / "streamlit_app"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from hf_volsurf.data.loaders import VolSurfaceDataLoader
from hf_volsurf.utils.vol_math import STRIKE_GRID, TENOR_ORDER, tenor_to_years
from utils.config import DB_PATH

st.set_page_config(page_title="Surface Explorer", layout="wide")
st.title("Surface Explorer")

with st.expander("What is this page?", expanded=False):
    st.markdown("""
**Purpose:** Browse 16 years of real SPX implied volatility surfaces interactively.

**Why it matters:** An implied volatility surface maps how the market prices risk across
different strikes (moneyness) and maturities (tenor). The shape encodes expectations about
future returns — skew reflects crash risk, term structure reflects vol regime expectations.

**What to look for:**
- **Skew** (left side higher than right): equity markets always price downside protection
  more than upside. The steeper the skew, the more the market fears a crash.
- **Term structure**: during crises, short-term vol spikes above long-term (inversion).
  In calm markets, long-term vol is typically higher (contango).
- **Level**: ATM 1m IV ranges from 8.8% (calm) to 77.5% (COVID crash peak).

**Quick picks** let you jump to key market events to see how the surface responded.

**Data source:** Marquee (Goldman Sachs), copied from the `rl_hedging_comparison` project.
4,134 complete daily surfaces, each with 104 grid points (8 tenors x 13 strikes).
""")


@st.cache_resource
def load_data():
    loader = VolSurfaceDataLoader(DB_PATH)
    grids, dates = loader.get_all_surface_grids()
    return loader, grids, dates


loader, all_grids, all_dates = load_data()
strikes = np.array(STRIKE_GRID)
tenors_years = np.array([tenor_to_years(t) for t in TENOR_ORDER])

# --- Sidebar ---
st.sidebar.header("Select Date")
date_str = st.sidebar.text_input("Date (YYYY-MM-DD)", value="2020-03-16")

# Quick-pick buttons
st.sidebar.markdown("**Quick picks:**")
quick_picks = {
    "COVID crash": "2020-03-16",
    "Volmageddon": "2018-02-05",
    "SVB collapse": "2023-03-10",
    "Low vol 2023": "2023-06-15",
    "Carry unwind": "2024-08-05",
}
for label, qdate in quick_picks.items():
    if st.sidebar.button(label, key=f"qp_{qdate}"):
        date_str = qdate

# --- Main content ---
grid = loader.get_surface_as_grid(date_str)

if grid is None:
    st.error(f"No data for {date_str}. Available: {all_dates[0]} to {all_dates[-1]}")
else:
    col1, col2 = st.columns([2, 1])

    with col1:
        X, Y = np.meshgrid(strikes, tenors_years)
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=grid,
            colorscale="Viridis",
            colorbar=dict(title="IV"),
        )])
        fig.update_layout(
            title=f"SPX Vol Surface — {date_str}",
            scene=dict(
                xaxis_title="Moneyness (K/S)",
                yaxis_title="Tenor (years)",
                zaxis_title="Implied Vol",
            ),
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        atm_idx = STRIKE_GRID.index(1.0)
        atm_1m = grid[0, atm_idx]
        atm_1y = grid[5, atm_idx]
        skew = grid[0, STRIKE_GRID.index(0.9)] - grid[0, STRIKE_GRID.index(1.1)]

        st.metric("ATM 1m IV", f"{atm_1m:.2%}")
        st.metric("ATM 1y IV", f"{atm_1y:.2%}")
        st.metric("1m Skew (90-110%)", f"{skew:.4f}")
        st.metric("Min IV", f"{grid.min():.4f}")
        st.metric("Max IV", f"{grid.max():.4f}")

    # Smile plot
    st.subheader("Smile by Tenor")
    import matplotlib.pyplot as plt

    fig_smile, ax = plt.subplots(figsize=(10, 4))
    for i, tenor in enumerate(TENOR_ORDER):
        ax.plot(strikes, grid[i], label=tenor, marker="o", markersize=3)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Vol")
    ax.legend(ncol=4, fontsize=8)
    ax.set_title(f"Volatility Smile — {date_str}")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_smile)
