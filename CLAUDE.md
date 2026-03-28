# CLAUDE.md — hugging-face-learning

**Hugging Face for Quantitative Finance: Volatility Surface Modeling**

Learning project exploring the Hugging Face ecosystem (Transformers, Diffusers,
Datasets, Gradio, Spaces) applied to implied volatility surface modeling.
Three ML approaches: FinBERT sentiment-informed vol, Transformer-based surface
prediction, and DDPM generative surfaces.

## IMPORTANT: Read Skills Before Any Work

**MANDATORY:** Before writing ANY code, read the skills listed below that match
the task. Do NOT skip skills. After completing work, run through each relevant
skill's checklist to verify compliance.

If the user says "check skills" — re-read all relevant skills and verify the
current code against their rules and checklists. Report any violations.

### Always read (every session)
- ~/skills/skills/project-scaffold/SKILL.md — directory structure, launch.json, Streamlit layout
- ~/skills/skills/env-setup/SKILL.md — Poetry detection, .env, credentials
- ~/skills/skills/git-workflow/SKILL.md — init vs upgrade, branching, commits

### Read when working with data
- ~/skills/skills/market-data/SKILL.md — DB schema, incremental updates, copy between projects
- ~/skills/skills/edav/SKILL.md — data quality, outliers, visualisation
- ~/skills/skills/experiment-logging/SKILL.md — log runs, model registry, compare versions

### Read when building ML models
- ~/skills/skills/keras3-pytorch/SKILL.md — Keras 3, train/val/test, callbacks, data pipeline
- ~/skills/skills/notebook-workflow/SKILL.md — notebook → module → Streamlit, README phases
- ~/skills/skills/experiment-workflow/SKILL.md — hyperparameter tuning, model selection
- ~/skills/skills/paper-replication/SKILL.md — NotebookLM → notebooks → src

### Read when writing quant/finance code
- ~/skills/skills/vol-and-curves/SKILL.md — vol interpolation, rate curves, estimators, 252/365
- ~/skills/skills/pricing/SKILL.md — BS, Greeks, Monte Carlo, PDE, payoffs
- ~/skills/skills/quant-patterns/SKILL.md — calibration, state normalisation, convergence
- ~/skills/skills/backtesting/SKILL.md — walk-forward, risk metrics, transaction costs

### Read when writing tests
- ~/skills/skills/testing-conventions/SKILL.md — staged testing, pytest, file discipline

### Optional
- ~/skills/skills/ci-cd/SKILL.md — GitHub Actions (add when project is mature)

## Quick Menu

| Command | What happens |
|---------|-------------|
| `init` | New project — interactive, step by step on main |
| `init auto` | New project — autonomous, works on a branch |
| `upgrade` | Existing project — interactive, step by step on a branch |
| `upgrade auto` | Existing project — autonomous, works on a branch |
| `check skills` | Re-read all relevant skills, verify code compliance, report violations |

### Useful Claude Code commands

| Command | What it does |
|---------|-------------|
| `Shift+Tab` | Cycle permission mode: default → acceptEdits → plan → auto |
| `/plan` | Enter plan mode — Claude analyses without changing anything |
| `/effort max` | Maximum thinking depth — use for planning and complex decisions |
| `/effort medium` | Balanced — use for normal coding |
| `/compact` | Compress conversation to free up tokens |
| `/clear` | Start fresh (resets conversation) |
| `@filename` | Reference a file directly in your prompt |
| `Ctrl+O` | Show Claude's thinking process |
| `/rc` | Enable Remote Control for phone/browser monitoring |

## Work Mode

When the user starts a session, detect the mode from their command:

- **Interactive** (`init` or `upgrade`) — ask before each step, user reviews and approves.
  Good for: new projects, complex upgrades, learning.

- **Autonomous** (`init auto` or `upgrade auto`) — work through the plan independently,
  commit progress, and report back when done or when stuck.
  Good for: well-defined tasks, background work, overnight runs.
  **Safety rule: autonomous mode ALWAYS works on a branch, never main.**

### How to start autonomous mode

```bash
# 1. Start Claude Code with permissions skipped
claude --dangerously-skip-permissions

# 2. Enable Remote Control so you can monitor from phone/browser
/rc

# 3. Tell Claude what to do (init or upgrade), say "autonomous"

# 4. Walk away — monitor from Claude app or claude.ai/code
#    Claude works on a branch, commits progress, stops if tests fail
```

### Autonomous mode rules

If the user says "run autonomously", "work in background", or "just do it":
1. Create a feature branch immediately (e.g. `feat/auto-<task-description>`)
2. Work through the plan, committing after each step
3. Run tests after each step — stop if tests fail
4. Update PROGRESS.md as you go
5. When done, report what was completed and what's left
6. Do NOT merge to main — leave that for the user to review

## Commands

### "init" — New Project

**Do NOT start scaffolding immediately.** First, have a conversation:

1. **Ask the user:**
   - What does this project do? (one sentence)
   - What type of work is it? (ML model, data pipeline, paper replication, quant tool, app, etc.)
   - What data sources will it use?
   - Are there any reference papers or existing projects to base this on?
   - Any specific packages or approaches already in mind?
   - **How do you want to work? (interactive or autonomous)**

2. **Save the answers into this CLAUDE.md immediately:**
   - Add the project name and description at the top of this file (under the `# CLAUDE.md` heading)
   - Fill in **Project Rules** with the decisions made
   - Fill in **Architecture** with the planned structure
   - Fill in **Current State** with the starting point
   - Add any constraints to **Do NOT**
   - Write the file now — do not wait until later

3. **Create PROGRESS.md** with the initial plan under "Next".

4. **Confirm the plan with the user** — show them the updated CLAUDE.md
   and ask "Does this look right? Ready to go?"

5. **Only then start scaffolding** — read project-scaffold and env-setup skills,
   check for existing Poetry environment, scaffold the directory structure,
   and work on `main`.

### "upgrade" — Existing Project

**Do NOT start coding immediately.** First, understand the objective:

1. **Ask the user:**
   - What needs to change? (e.g. "migrate from CSV to DB", "add Heston model", "refactor training loop")
   - Why? (e.g. "CSV is too slow", "need stochastic vol", "code is messy")
   - Are there any reference materials? (papers, other projects, docs)
   - What should NOT break? (existing notebooks, trained models, API interfaces)
   - What does success look like? (e.g. "same results, faster loading", "new model calibrates with RMSE < 0.01")
   - **How do you want to work? (interactive or autonomous)**

2. **Save the answers into CLAUDE.md immediately:**
   - Add the upgrade objective to **Current State**
   - Add any new rules to **Project Rules**
   - Add constraints to **Do NOT** (e.g. "do not require model retraining")
   - Update **Architecture** if the structure will change
   - Write the file now — do not wait until later

3. **Update PROGRESS.md** — add the upgrade objective to "In Progress".

4. **Present the plan** — show the user:
   - What branch will be created
   - What steps will be taken (in order)
   - What will be tested after each step
   - What the merge criteria are

5. **Confirm with the user** — "Does this plan look right? Ready to proceed?"

6. **Only then start working** — read upgrade-repo and git-workflow skills,
   create the branch, and execute step by step.

## Keeping this file and PROGRESS.md up to date

**CLAUDE.md** is the source of truth for project configuration.
**PROGRESS.md** is the source of truth for what's been done and what's next.

### At the start of every session

Before doing any work, quickly review this CLAUDE.md and check:
- Does **Architecture** still match the actual directory structure?
- Does **Current State** reflect what's actually happening?
- Are there **Project Rules** that have been superseded by new decisions?
- Are there missing rules or constraints from recent work?

If anything is stale, update it silently. If something significant changed,
mention it to the user: "I noticed the architecture section was outdated —
updated it to reflect the new data pipeline."

Also read PROGRESS.md and check what's in "In Progress" and "Next" to
understand where the project stands.

### During a session

When the user asks to change the project, update both files:

**CLAUDE.md updates:**
- Add new project rules under **Project Rules** when decisions are made
- Add constraints under **Do NOT** when the user says to avoid something
- Update **Architecture** when the project structure changes
- Update **Current State** when branches, active work, or known issues change

**PROGRESS.md updates:**
- Move completed work from "In Progress" to "Done"
- Add new work to "In Progress" when starting
- Add planned work to "Next" when discussed
- Record key decisions under "Decisions" with date and reasoning

The user should never need to edit these files by hand. If they want to change
something, they tell Claude and Claude updates the code and both files.

## Project Rules

1. This is a **learning/exploration** project — favour clarity over production polish.
2. Each ML approach (FinBERT, Transformer, Diffusion) lives in its own module under `src/`.
3. Notebooks are the primary exploration tool — promote to `src/` only when validated.
4. Use HF libraries as the primary API surface (`transformers`, `diffusers`, `datasets`, `gradio`).
5. PyTorch is the backend (via Keras 3 where applicable).
6. Use **real SPX data** — copy from `rl_hedging_comparison/data/db/rl_hedging_data.db` via market-data skill's `copy_tables()`.
6b. Synthetic data (Black-Scholes generated surfaces) as fallback / augmentation only.
7. All vol surfaces use **moneyness** (K/S) on x-axis, not raw strike.
8. Day-count convention: 252 trading days/year for realized vol, 365 for implied vol term structure.
9. Poetry for dependency management, `src/` layout.
10. Gradio app for interactive demo; target HF Spaces deployment.

## Architecture

```
hugging-face-learning/
├── CLAUDE.md
├── PROGRESS.md
├── README.md
├── pyproject.toml
├── .env.example
├── .pre-commit-config.yaml
├── .gitignore
├── .vscode/
│   └── launch.json
├── src/
│   └── hf_volsurf/
│       ├── __init__.py
│       ├── data/               # Data loading, HF Dataset wrappers
│       │   ├── __init__.py
│       │   ├── loaders.py      # VolSurfaceDataLoader (SQLite → grids)
│       │   └── hf_dataset.py   # HF Datasets integration
│       ├── models/             # ML model definitions
│       │   ├── __init__.py
│       │   ├── finbert.py      # FinBERT sentiment pipeline (frozen)
│       │   ├── transformer.py  # Time-series Transformer for vol
│       │   └── diffusion.py    # DDPM for generative surfaces
│       ├── evaluation/         # Metrics, no-arb checks
│       │   ├── __init__.py
│       │   └── metrics.py      # RMSE, MAE, arb violation rate
│       └── utils/
│           ├── __init__.py
│           └── vol_math.py     # BS formula, tenor conversion, arb checks
├── streamlit_app/              # Dashboard (NOT inside src/)
│   ├── app.py                  # Main entry point
│   ├── utils/
│   │   └── config.py           # Shared paths
│   └── pages/
│       ├── 1_Surface_Explorer.py
│       ├── 2_DDPM_Generator.py
│       ├── 3_FinBERT_Sentiment.py
│       └── 4_Data_Summary.py
├── notebooks/
│   ├── 01_hf_ecosystem_tour.ipynb
│   ├── 02_real_vol_surface_eda.ipynb
│   ├── 03_finbert_sentiment.ipynb
│   ├── 04_transformer_vol.ipynb
│   ├── 05_diffusion_vol.ipynb
│   └── 06_gradio_demo.ipynb
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── scripts/
├── output/
└── data/
```

## Current State

- **Branch:** main
- **Repo:** https://github.com/aesp77/hugging-face
- **Status:** Phase 3 complete — notebooks done, modules extracted, Streamlit app running
- **Next step:** Improve Transformer (predict changes), conditional DDPM, HF Spaces deployment

## Do NOT

- Do not use `pip install` — use `poetry add`
- Do not build production-grade pipelines — this is for learning HF
- Do not skip no-arbitrage checks when evaluating generated surfaces
- Do not hardcode file paths — use `pathlib.Path` relative to project root
- Do not train on real market data without first validating on synthetic data
- Do not commit `.env`, data files, or model checkpoints

Create `PROGRESS.md` in the project root during init. Keep it updated
as work progresses. Format:

```markdown
# Progress

## Done
<!-- Completed milestones — add date and one-line summary -->

## In Progress
<!-- What's currently being worked on -->

## Next
<!-- Planned but not started — in priority order -->

## Decisions
<!-- Key decisions made, with date and reasoning -->
<!-- e.g. 2026-03-21: Chose Optuna over Keras Tuner — better pruning support -->

## References
<!-- Papers, links, other projects informing this work -->
```
