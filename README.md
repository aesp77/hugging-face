# Hugging Face for Quantitative Finance: Volatility Surface Modeling

## Motivation

Implied volatility surfaces are central to derivatives pricing, risk management,
and trading. A vol surface maps implied volatility as a function of strike price
(or moneyness) and time to expiration. Traditional approaches (SABR, SVI, Heston
calibration) fit parametric models to market quotes. This project explores
whether modern ML architectures — available through the Hugging Face ecosystem —
can learn, interpolate, and generate volatility surfaces more flexibly.

The goal is not to replace quantitative models but to **test what the HF stack
can do** when pointed at a real finance problem.

## Background

### Volatility Surfaces

An implied volatility surface $\sigma(K, T)$ is extracted from observed option
prices by inverting the Black-Scholes formula. Key properties:

- **Smile/Skew:** For a fixed expiry, IV typically decreases with strike (equity
  skew) or forms a U-shape (smile). This reflects fat tails and jump risk not
  captured by log-normal assumptions.
- **Term Structure:** At-the-money IV varies with expiry, reflecting the market's
  view of future realized volatility regimes.
- **No-Arbitrage Constraints:** A valid surface must satisfy calendar spread and
  butterfly spread constraints — no negative forward variances, no negative
  butterfly prices.
- **Dynamics:** The surface moves through time as markets reprice risk. Modeling
  these dynamics is harder than fitting a single snapshot.

### Traditional Approaches

| Method | Idea | Limitation |
|--------|------|------------|
| **SVI** (Gatheral 2004) | Parameterize total variance as a function of log-moneyness | 5 params per slice, no cross-slice consistency |
| **SABR** (Hagan et al. 2002) | Stochastic vol + stochastic forward | Calibration per expiry, can break at wings |
| **Heston** (1993) | Closed-form option prices from stochastic vol SDE | Joint calibration is slow, limited smile flexibility |
| **Local Vol** (Dupire 1994) | Back out a deterministic vol function from the full surface | Requires dense, smooth market data |

All of these require careful numerical work and domain expertise. ML approaches
offer an alternative: learn the mapping from inputs to surface directly from data.

### Why Hugging Face?

Hugging Face provides a unified platform connecting:

1. **Datasets** — Standardized data loading (`load_dataset`), versioning, and
   community-shared financial datasets.
2. **Models** — Pre-trained architectures (Transformers, Diffusion models) that
   can be fine-tuned on domain-specific data.
3. **Training** — The `Trainer` API handles training loops, evaluation, logging,
   and checkpointing.
4. **Deployment** — Gradio for interactive UIs; Spaces for hosting.

This project uses HF as the backbone to explore three ML approaches to vol
surfaces.

## Approaches

### 1. Sentiment-Informed Volatility (FinBERT)

**Hypothesis:** Financial news sentiment is a leading indicator of vol regime
changes. FinBERT (Araci 2019) is a BERT model fine-tuned on financial text
(analyst reports, earnings calls, financial news) to classify sentiment as
positive, negative, or neutral.

**Pipeline:**
- Load financial news corpus via HF Datasets
- Run FinBERT inference to extract sentiment scores
- Correlate sentiment shifts with subsequent changes in ATM implied volatility
- Test whether sentiment features improve vol forecasting models

**HF Components:** `AutoModelForSequenceClassification`, `AutoTokenizer`,
`pipeline("sentiment-analysis")`

### 2. Time-Series Transformers for Vol Prediction

**Hypothesis:** Transformer architectures (attention over temporal sequences)
can capture the complex, non-linear dynamics of vol surface evolution better
than traditional time-series models (GARCH, HAR).

**Pipeline:**
- Construct a dataset of historical vol surface snapshots (grid of IV values
  across strikes and expiries, daily frequency)
- Tokenize/embed the numeric surface data into vector representations
- Train a Transformer encoder-decoder to predict the next day's surface given
  a lookback window
- Compare against GARCH(1,1) and HAR-RV baselines

**HF Components:** `AutoModel`, custom `Dataset` class, `Trainer`,
`TrainingArguments`

### 3. Generative Vol Surfaces (Diffusion Models)

**Hypothesis:** Denoising Diffusion Probabilistic Models (DDPMs) can learn the
distribution of valid vol surfaces and generate new, arbitrage-free surfaces
from noise — similar to how image diffusion models generate realistic faces.

**Pipeline:**
- Represent each vol surface snapshot as a 2D image (strike x expiry grid,
  IV values as pixel intensities)
- Train a DDPM (via HF Diffusers library) to denoise random noise into
  coherent surfaces
- Evaluate generated surfaces for no-arbitrage violations (calendar spreads,
  butterfly spreads)
- Use conditional generation to produce surfaces given market state inputs

**HF Components:** `DDPMScheduler`, `DDPMPipeline`, `UNet2DModel` from
`diffusers`

## Deployment: Gradio + HF Spaces

Once models are trained, the project builds an interactive demo:

- **Inputs:** Sliders for spot price, risk-free rate, time-to-maturity range;
  text box for news headlines (FinBERT sentiment)
- **Outputs:** 3D surface plot (matplotlib/plotly), predicted ATM vol level,
  sentiment score breakdown
- **Hosting:** Deployed as a Hugging Face Space with GPU support for diffusion
  inference

**HF Components:** `gradio.Interface`, `gr.Slider`, `gr.Image`, `gr.Label`,
HF Spaces deployment

## Data Sources

| Source | What | Format |
|--------|------|--------|
| CBOE / OptionMetrics | SPX/SPY option chains with IV | CSV / DB |
| Yahoo Finance (yfinance) | Historical prices, basic option chains | API |
| HF Datasets Hub | Community financial datasets | Arrow / Parquet |
| Financial news APIs | Headlines for FinBERT sentiment | JSON / text |
| Synthetic (Black-Scholes) | Generated surfaces for diffusion training | NumPy arrays |

## Tech Stack

| Layer | Tool |
|-------|------|
| ML Framework | PyTorch (via Keras 3 backend) |
| NLP Models | Hugging Face Transformers (FinBERT, custom) |
| Generative Models | Hugging Face Diffusers (DDPM, UNet2D) |
| Data Loading | Hugging Face Datasets, pandas, numpy |
| Visualization | matplotlib, plotly (3D surfaces) |
| Interactive UI | Gradio |
| Deployment | Hugging Face Spaces |
| Experiment Tracking | TBD (Weights & Biases or TensorBoard) |
| Package Management | Poetry |

## References

- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives.* Global Derivatives & Risk Management.
- Hagan, P. et al. (2002). *Managing Smile Risk.* Wilmott Magazine.
- Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility.* Review of Financial Studies.
- Dupire, B. (1994). *Pricing with a Smile.* Risk Magazine.
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models.* arXiv:1908.10063.
- Ho, J., Jain, A., Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Hugging Face Course (2024). *https://huggingface.co/learn*
