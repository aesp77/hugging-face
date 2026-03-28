"""Page 4: Database overview and data coverage."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(_project_root) / "src"))
sys.path.insert(0, str(Path(_project_root) / "streamlit_app"))

import json

import numpy as np
import streamlit as st

from hf_volsurf.data.loaders import VolSurfaceDataLoader
from hf_volsurf.evaluation.metrics import arbitrage_violation_rate
from hf_volsurf.utils.vol_math import STRIKE_GRID, TENOR_ORDER
from utils.config import DB_PATH, OUTPUT_DIR

st.set_page_config(page_title="Data Summary", layout="wide")
st.title("Data Summary")

with st.expander("What is this page?", expanded=False):
    st.markdown("""
**Purpose:** Overview of the database powering this app — row counts, date coverage,
train/val/test split, and no-arbitrage compliance of the real market data.

**Data origin:** All data was copied from the `rl_hedging_comparison` project's SQLite database
(58 MB), which in turn was sourced from:
- **Marquee (Goldman Sachs)** — vol surfaces (429k grid points, 2010-2026)
- **PSC / FirstRate** — SPX spot prices (4k daily, 2010-2026)
- **Bloomberg** — OIS rates and dividend yields

**Train / Val / Test split:** Chronological (no shuffling) to prevent look-ahead bias:
- **Train** (2010-2021): 3,113 surfaces — model training
- **Val** (2022-2023): 512 surfaces — hyperparameter tuning, early stopping
- **Test** (2024-2026): 509 surfaces — final evaluation, never seen during training

**No-arbitrage compliance:** Real Marquee surfaces are interpolated, not arbitrage-free
by construction. About 17% violate calendar-spread constraints (total variance not
monotonically increasing in maturity). Butterfly constraints (strike convexity) are
always satisfied. These rates set the benchmark for the DDPM generative model.

**Grid definition:** Each surface has 8 tenors (1m to 2y) x 13 strikes (70% to 130%
moneyness) = 104 implied volatility values per trading day.
""")



@st.cache_resource
def load_summary():
    loader = VolSurfaceDataLoader(DB_PATH)
    summary = loader.get_data_summary()
    grids, dates = loader.get_all_surface_grids()
    return summary, grids, dates


summary, grids, dates = load_summary()

# --- Table overview ---
st.subheader("Database Tables")
import pandas as pd

table_df = pd.DataFrame([
    {"Table": t, "Rows": f"{info['rows']:,}", "From": info["min_date"], "To": info["max_date"]}
    for t, info in summary.items()
])
st.dataframe(table_df, use_container_width=True, hide_index=True)

st.metric("Database", str(DB_PATH.name))
st.metric("Total surfaces (complete grids)", f"{len(dates):,}")
st.metric("Grid shape", f"{len(TENOR_ORDER)} tenors × {len(STRIKE_GRID)} strikes = {len(TENOR_ORDER) * len(STRIKE_GRID)} points")

# --- Split info ---
st.subheader("Train / Val / Test Split")
split_path = OUTPUT_DIR / "02_split_info.json"
if split_path.exists():
    with open(split_path) as f:
        split_info = json.load(f)
    col1, col2, col3 = st.columns(3)
    col1.metric("Train", f"{split_info['n_train']:,} surfaces", "2010 — 2021")
    col2.metric("Val", f"{split_info['n_val']:,} surfaces", "2022 — 2023")
    col3.metric("Test", f"{split_info['n_test']:,} surfaces", "2024 — 2026")
else:
    st.info("Run NB 02 to generate split info.")

# --- Arbitrage checks ---
st.subheader("No-Arbitrage Compliance")
with st.spinner("Checking arbitrage on all surfaces..."):
    arb = arbitrage_violation_rate(grids)
col1, col2 = st.columns(2)
col1.metric("Calendar arb free", f"{arb['calendar_free']:,}/{arb['total']:,} ({arb['calendar_free_pct']:.1f}%)")
col2.metric("Butterfly arb free", f"{arb['butterfly_free']:,}/{arb['total']:,} ({arb['butterfly_free_pct']:.1f}%)")

# --- Strike and tenor grids ---
st.subheader("Grid Definition")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Tenors:**")
    st.code(", ".join(TENOR_ORDER))
with col2:
    st.markdown("**Strikes (moneyness K/S):**")
    st.code(", ".join(f"{s:.2f}" for s in STRIKE_GRID))
