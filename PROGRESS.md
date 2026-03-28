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
- 2026-03-28: **Phase 2 complete** — all 6 exploration notebooks
  - **NB 01** — HF ecosystem tour: Datasets API, HfApi, FinBERT pipeline (3 lines of code)
  - **NB 02** — Vol surface EDA: 4,134 surfaces, ATM 1m IV [8.8%, 77.5%], VIX-ATM r=0.81, 16.8% calendar arb violations in real data, split 3113/512/509
  - **NB 03** — FinBERT sentiment: scored 24 headlines, sentiment-vol correlation r=-0.575, negative sentiment aligns with vol spikes
  - **NB 04** — Transformer vol prediction: 208K params, 10 epochs, naive baseline wins (vol surfaces are very persistent — yesterday IS a strong predictor)
  - **NB 05** — DDPM generative surfaces: 651K params, 30 epochs, generated 100 surfaces, 76% calendar arb free (matches real 78%), 100% butterfly arb free, mean IV diff 0.005
  - **NB 06** — Gradio demo: 3-tab app (Explorer, DDPM Generator, FinBERT), all smoke-tested

## In Progress
<!-- Nothing -->

## Next
1. **Phase 3: Promote to src/** — extract validated notebook code into modules
2. **Phase 4: Gradio app + HF Spaces** — deployment
3. **Improve Transformer** — try: predict vol changes (not levels), add VIX/spot as features, increase lookback, experiment with Chronos-T5
4. **Conditional DDPM** — condition on VIX level or market regime

## Decisions
- 2026-03-28: Package name `hf_volsurf` — short, descriptive, avoids conflict with `huggingface` namespace
- 2026-03-28: **Real data first** — copied from rl_hedging_comparison (429k vol surface points, 2010-2026) instead of synthetic
- 2026-03-28: Keep rl_hedging table names (spx_vol_surface etc.) — avoids schema translation
- 2026-03-28: Grid shape (8, 13) verified consistent across all 4k+ dates
- 2026-03-28: Train/val/test split: 2010-2021 / 2022-2023 / 2024-2026 (chronological, no shuffle)
- 2026-03-28: Three parallel ML tracks (FinBERT, Transformer, Diffusion) — each explores a different HF library
- 2026-03-28: Transformer underperforms naive — vol surfaces are highly persistent, need to predict changes not levels
- 2026-03-28: DDPM works surprisingly well — matches real data arbitrage rates, mean IV error ~0.5%

## References
- Hugging Face Course: https://huggingface.co/learn
- FinBERT: arXiv:1908.10063
- DDPM: Ho et al. (2020), NeurIPS
- Gatheral SVI (2004) — vol surface parameterization baseline
- Heston (1993) — stochastic vol model baseline
- Source DB: `rl_hedging_comparison/data/db/rl_hedging_data.db`
- Vol surface utilities: `spx_lookback_pricer/src/.../data/vol_surface.py`
