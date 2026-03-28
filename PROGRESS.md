# Progress

## Done
- 2026-03-28: Project initialized — CLAUDE.md, README.md (scientific background), directory scaffold
- 2026-03-28: **Phase 1 complete** — data infrastructure
  - Copied 429,936 vol surface rows + spot/div/OIS/VIX from rl_hedging_comparison
  - DB at `data/db/hf_volsurf.db` (69.4 MB)
  - `VolSurfaceDataLoader` with grid access (8 tenors x 13 strikes = 104 pts/day)
  - HF Dataset wrappers (pointwise + image format)
  - Vol math utilities (BS, tenor conversion, arb checks, normalization)
  - 18/18 unit tests passing
- 2026-03-28: **Phase 2 complete** — all 6 exploration notebooks with markdown commentary
  - NB 01: HF ecosystem tour — Datasets API, HfApi, FinBERT pipeline
  - NB 02: Vol surface EDA — 4,134 surfaces, VIX-ATM r=0.81, 16.8% calendar arb, split 3113/512/509
  - NB 03: FinBERT sentiment — r=-0.575 sentiment-vol correlation
  - NB 04: Transformer — naive baseline wins (surfaces too persistent for level prediction)
  - NB 05: DDPM — 80% calendar arb free (matches real 78%), mean IV diff 0.002
  - NB 06: Gradio demo — 3-tab prototype smoke-tested
- 2026-03-28: **Phase 2b complete** — extracted modules to src/
  - `models/finbert.py` — FinBERTSentiment wrapper with score() and score_df()
  - `models/transformer.py` — VolSurfaceTransformer + VolSurfaceWindowDataset
  - `models/diffusion.py` — VolSurfaceDDPM with generate() and weight save/load
  - `evaluation/metrics.py` — RMSE, MAE, naive baseline, arb violation rate
- 2026-03-28: **Phase 3 complete** — Streamlit app
  - 4-page app: Surface Explorer, DDPM Generator, FinBERT Sentiment, Data Summary
  - All pages import from `src/hf_volsurf/`, no re-implemented logic
  - launch.json updated for monorepo
- 2026-03-28: Pushed to https://github.com/aesp77/hugging-face

## In Progress
<!-- Nothing -->

## Next
1. **Improve Transformer** — predict vol *changes* not levels, add VIX/spot as conditioning features
2. **Conditional DDPM** — condition on VIX level or market regime
3. **HF Spaces deployment** — push Gradio version to HF Spaces for public demo
4. **README Phase 2** — add setup/run/structure sections now that Streamlit app is ready

## Decisions
- 2026-03-28: Package name `hf_volsurf`
- 2026-03-28: **Real data first** — copied from rl_hedging_comparison instead of synthetic
- 2026-03-28: Keep rl_hedging table names (spx_vol_surface etc.)
- 2026-03-28: Grid shape (8, 13) verified consistent across all 4k+ dates
- 2026-03-28: Train/val/test split: 2010-2021 / 2022-2023 / 2024-2026 (chronological)
- 2026-03-28: Transformer underperforms naive — vol surfaces are highly persistent, need to predict changes
- 2026-03-28: DDPM is the star — matches real data arb rates, mean IV error 0.2%
- 2026-03-28: Streamlit for local dashboard (per skills), Gradio/HF Spaces for public demo

## References
- Hugging Face Course: https://huggingface.co/learn
- FinBERT: arXiv:1908.10063
- DDPM: Ho et al. (2020), NeurIPS
- Gatheral SVI (2004) — vol surface parameterization baseline
- Heston (1993) — stochastic vol model baseline
- Source DB: `rl_hedging_comparison/data/db/rl_hedging_data.db`
- Vol surface utilities: `spx_lookback_pricer/src/.../data/vol_surface.py`
- Repo: https://github.com/aesp77/hugging-face
