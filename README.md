# ai-compute-users

Estimates of the compute quantities used by major AI players: Monte Carlo
models of frontier labs' compute (Google DeepMind, Meta Superintelligence
Labs, OpenAI, Anthropic) in H100-equivalents, plus Alphabet-level activity
estimates.

Split out of [epoch-research/ai-chip-counts](https://github.com/epoch-research/ai-chip-counts)
(its `ai-lab-compute/` directory, history preserved).

## Layout

- `frontier_lab_compute_model.py` — consolidated end-2025 models for the four
  labs, plus end-2024 backcasts for Google DeepMind and Meta AI. The priors
  live in `lab_model_params.csv` (loaded via `lab_compute_utils.py`).
- `*_compute_model` / `*_monte_carlo` / `*_backcast` notebooks — the canonical
  per-lab walkthroughs, jupytext-paired (`.ipynb` ↔ `.py`; edit the `.py` and
  run `sync_notebooks.sh` or `jupytext --sync --execute`).
- `lab_compute_tables/` — exports the results as tables and a consolidated
  walkthrough page.
- `data/` — hand-maintained model inputs (see `data/README.md`).
- `epoch_data.py` — chip fleet data (Nvidia per-owner fleets, TPU and AMD
  cumulative sales) fetched at runtime from the
  [Epoch AI data hub](https://epoch.ai/data) (AI Chip Sales and AI Chip
  Owners), cached one download per day under `.cache/`.

## Setup

Python 3.11 with `pip install -r requirements.txt`. Runs need network access
for the Epoch data hub fetch.
