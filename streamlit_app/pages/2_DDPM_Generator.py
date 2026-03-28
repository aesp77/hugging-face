"""Page 2: Generate vol surfaces with DDPM."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(_project_root) / "src"))
sys.path.insert(0, str(Path(_project_root) / "streamlit_app"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from hf_volsurf.data.loaders import VolSurfaceDataLoader
from hf_volsurf.evaluation.metrics import arbitrage_violation_rate
from hf_volsurf.models.diffusion import DDPMConfig, VolSurfaceDDPM
from hf_volsurf.utils.vol_math import STRIKE_GRID, TENOR_ORDER, tenor_to_years
from utils.config import DB_PATH, OUTPUT_DIR

st.set_page_config(page_title="DDPM Generator", layout="wide")
st.title("DDPM Generative Vol Surfaces")

with st.expander("What is this page?", expanded=False):
    st.markdown("""
**Purpose:** Generate new, realistic implied volatility surfaces from pure random noise
using a trained **Denoising Diffusion Probabilistic Model** (DDPM) from the HF `diffusers` library.

**How it works:** The DDPM starts from Gaussian noise and iteratively denoises it over 1,000
steps into a coherent vol surface. The UNet2D model (651K parameters) was trained on 3,113
daily SPX surfaces (2010-2021) to learn what a "valid" surface looks like.

**Why it matters:** This is the standout result of the project. The model learns the
statistical structure of vol surfaces — including **no-arbitrage constraints** — purely from
data, with no explicit supervision:

| Constraint | Generated | Real data |
|-----------|-----------|-----------|
| Calendar arb free | ~80% | 78% |
| Butterfly arb free | 100% | 100% |
| Mean IV difference | 0.2% | — |

**Applications:**
- **Data augmentation** — generate additional surfaces for training other models
- **Scenario generation** — sample "what-if" surfaces for risk management
- **Stress testing** — generate extreme but plausible surfaces

**HF library:** `diffusers.UNet2DModel` + `diffusers.DDPMScheduler`
""")



@st.cache_resource
def load_ddpm():
    loader = VolSurfaceDataLoader(DB_PATH)
    grids, _ = loader.get_all_surface_grids()
    ddpm = VolSurfaceDDPM()
    ddpm.set_normalization(float(grids.min()), float(grids.max()))
    weights_path = OUTPUT_DIR / "05_ddpm_model.pt"
    if weights_path.exists():
        ddpm.load_weights(weights_path)
    else:
        st.warning("DDPM weights not found — run NB 05 first.")
    return ddpm


ddpm = load_ddpm()
strikes = np.array(STRIKE_GRID)
tenors_years = np.array([tenor_to_years(t) for t in TENOR_ORDER])

# --- Sidebar ---
st.sidebar.header("Generation Settings")
n_surfaces = st.sidebar.slider("Number of surfaces", 1, 20, 5)

if st.sidebar.button("Generate", type="primary"):
    with st.spinner(f"Generating {n_surfaces} surfaces (1000 denoising steps)..."):
        generated = ddpm.generate(n_surfaces)

    st.session_state["generated"] = generated

if "generated" not in st.session_state:
    st.info("Click **Generate** to create vol surfaces from noise.")
    st.stop()

generated = st.session_state["generated"]

# --- Results ---
col1, col2 = st.columns([2, 1])

with col1:
    # Show first surface as 3D
    X, Y = np.meshgrid(strikes, tenors_years)
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=generated[0],
        colorscale="Viridis",
        colorbar=dict(title="IV"),
    )])
    fig.update_layout(
        title="Generated Surface (sample 1)",
        scene=dict(
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Tenor (years)",
            zaxis_title="Implied Vol",
        ),
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    arb = arbitrage_violation_rate(generated)
    st.metric("Surfaces generated", len(generated))
    st.metric("Calendar arb free", f"{arb['calendar_free']}/{arb['total']} ({arb['calendar_free_pct']:.0f}%)")
    st.metric("Butterfly arb free", f"{arb['butterfly_free']}/{arb['total']} ({arb['butterfly_free_pct']:.0f}%)")
    st.metric("IV range", f"[{generated.min():.4f}, {generated.max():.4f}]")

# Heatmap grid of all generated surfaces
st.subheader("All Generated Surfaces")
n_cols = min(5, len(generated))
cols = st.columns(n_cols)
import matplotlib.pyplot as plt

for i, col in enumerate(cols):
    if i < len(generated):
        fig_hm, ax = plt.subplots(figsize=(3, 2))
        ax.imshow(generated[i], aspect="auto", cmap="viridis", vmin=0.05, vmax=0.6)
        ax.set_title(f"#{i+1}", fontsize=9)
        ax.set_yticks(range(8))
        ax.set_yticklabels(TENOR_ORDER, fontsize=6)
        ax.set_xticks([])
        col.pyplot(fig_hm)
