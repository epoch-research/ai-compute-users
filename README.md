# ai-compute-users

Estimates of the compute quantities used by major AI players: Monte Carlo
models of frontier labs' compute (Google DeepMind, Meta Superintelligence
Labs, OpenAI, Anthropic, SpaceXAI) in H100-equivalents (H100e), plus
Alphabet-level activity estimates.

Split out of [epoch-research/ai-chip-counts](https://github.com/epoch-research/ai-chip-counts)
(its `ai-lab-compute/` directory, history preserved).

## Research notebooks vs. the estimate script

The repo has two layers that share one set of priors:

- **Research notebooks** (`notebooks/`) — the canonical per-lab walkthroughs.
  Each notebook develops one model in full: the reasoning behind every input,
  sources, intermediate charts, and sensitivity sweeps. They are
  jupytext-paired (`.ipynb` ↔ `.py`): edit the `.py`, then run
  `notebooks/sync_notebooks.sh --execute` (or
  `python3.11 -m jupytext --sync --execute notebooks/<name>.py`) so outputs
  land in the `.ipynb`. Commit both files.
- **The estimate script** (`frontier_lab_compute_model.py`, repo root) — the
  consolidated, importable versions of the same models: end-2025 estimates for
  the five labs, the end-2024 backcasts for Google DeepMind, Meta, and
  Anthropic, and the SpaceXAI model (end-2024, end-2025, and mid-2026,
  anchored on Epoch's Colossus data-center capacity estimates). It
  restates no judgment priors — like the notebooks, it loads them from
  **`lab_model_params.csv`** (via `lab_compute_utils.load_lab_params()`), the
  single source of truth for the model priors. Change a prior in the sheet and
  both layers pick it up.

The notebooks:

| Notebook | Estimates |
|----------|-----------|
| `deepmind_compute_model` | Google DeepMind compute, end-2025 |
| `msl_compute_model` | Meta Superintelligence Labs compute, end-2025 |
| `openai_power_model` | OpenAI compute at year-ends 2023–2025 (power-based, mainline) |
| `anthropic_power_2025` | Anthropic compute, end-2025 (power-based, mainline) |
| `anthropic_cloud_spend_2024` | Anthropic compute, end-2024 (canonical) + end-2025 cloud-spend cross-check |
| `openai_cloud_spend` | OpenAI compute 2024–2025 via cloud spend (cross-check) |
| `alphabet_level_activities_model` | Alphabet-level activity estimates |
| `lab_2024_backcasts` | End-2024 backcasts across the four labs |
| `spacexai_compute_model` | SpaceXAI compute, end-2024 / end-2025 / mid-2026 (Colossus-anchored) |

Supporting modules (repo root): `lab_compute_utils.py` (prior loader, fleet
buildout helper), `epoch_data.py` (chip fleet data — Nvidia per-owner fleets,
TPU and AMD cumulative sales — plus data-center capacity timelines, fetched at
runtime from the [Epoch AI data hub](https://epoch.ai/data), cached one
download per day under `.cache/`), and `data/` (hand-maintained inputs; see
`data/README.md`).

## Table exports

`generate_lab_compute_tables.py` (repo root) turns the estimate script's
Monte Carlos into export tables: point-in-time total compute per lab in H100e,
with P5 / median / P95 uncertainty. It holds no model structure of its own — it runs the models
from `frontier_lab_compute_model.py` and shapes their sample arrays.

### Usage

```python
from generate_lab_compute_tables import get_all_tables

tables = get_all_tables()          # dict of two DataFrames
tables["year_end_by_lab"]          # (end-of-year × lab) headline estimates
tables["intermediates_by_lab"]     # the quantities behind each estimate
```

Or the individual getters:

```python
from generate_lab_compute_tables import get_year_end_by_lab, get_intermediates_by_lab

df = get_year_end_by_lab()
```

From the command line (writes both `data/lab_compute_*.csv` files):

```
python3.11 generate_lab_compute_tables.py
```

Results are deterministic: each lab model reseeds 42 internally so it
reproduces its canonical notebook run — hence no seed parameter.

### Schema — `year_end_by_lab`

| Column | Type | Notes |
|--------|------|-------|
| `Name` | str | `"{Lab} end-{Year}"`, e.g. `"OpenAI end-2025"` |
| `Lab` | str | `Google DeepMind` / `Meta Superintelligence Labs` / `OpenAI` / `Anthropic` |
| `Year` | int | Calendar year of the snapshot |
| `Date` | str | `YYYY-12-31` — the point in time the estimate refers to |
| `h100e_p5` / `h100e_med` / `h100e_p95` | float | Percentiles of total H100e |
| `Notes` | str | Generation timestamp |

H100e converts each chip at its dense 8-bit peak FLOP/s divided by the H100's
1979 TFLOP/s. Estimates cover compute **rented or used** by each lab (not
owned), at the stated moment in time — they are operational-stock snapshots,
not flows, so consecutive years must not be summed.

The 2024 rows for Google DeepMind, Meta, and Anthropic are backcasts. They
keep the same `Lab` labels for continuity, but "the lab" in 2024 means
*frontier-AI compute at the company*: Meta Superintelligence Labs did not
exist in 2024 (its predecessor was Meta AI / GenAI plus FAIR), and the
backcast share priors are for those predecessor scopes — see
`notebooks/lab_2024_backcasts.ipynb`. Anthropic's 2024 row converts reported
cloud spend at 2024 prices (`notebooks/anthropic_cloud_spend_2024.ipynb`).

### Schema — `intermediates_by_lab`

How each lab's final distribution is computed: one row per intermediate
quantity (owned fleets, deployment ratios, shares, power, chip counts, ...),
in model order. The traces come from `MODEL_STEPS` in
`frontier_lab_compute_model.py` — each model records its steps as pure
bookkeeping, so adding a `step(...)` entry there is all it takes to extend
this table (and the walkthrough page below).

| Column | Type | Notes |
|--------|------|-------|
| `Name` | str | `"{Lab} end-{Year} · {Label}"` |
| `Lab` / `Year` | str / int | Same conventions as `year_end_by_lab`; one trace per modelled (lab, year) snapshot |
| `Step` | int | 1-based position in the model's computation |
| `Variable` | str | Machine name; sheet-prior steps match their row name in `lab_model_params.csv` |
| `Label` | str | Human-readable name |
| `Kind` | str | `input` (sampled prior) / `constant` (fixed scalar) / `derived` / `final` |
| `Units` | str | `H100e`, `MW`, `share`, `ratio`, `quarters`, `chips`, `USD B/yr`, ... |
| `Expression` | str | For derived/final rows: how the step combines earlier ones, by `Variable` name |
| `value_p5` / `value_med` / `value_p95` | float | Percentiles in the row's own units; constants repeat the same value |
| `Notes` | str | Generation timestamp |

Each snapshot's `final` row equals its `year_end_by_lab` row (tested).

### Coverage

| Lab | Snapshots |
|-----|-----------|
| OpenAI | 2023, 2024, 2025 |
| Anthropic | 2024, 2025 |
| Google DeepMind | 2024, 2025 |
| Meta Superintelligence Labs | 2024, 2025 |
| SpaceXAI | 2024, 2025, mid-2026 |

OpenAI's power model yields a snapshot per disclosed year-end; DeepMind, Meta,
and Anthropic add end-2024 backcast models (Anthropic's converts reported
cloud spend at 2024 prices; an experimental power-model backcast stays
notebook-only in `notebooks/anthropic_cloud_spend_2024.ipynb`).
Rows are omitted (not zero-padded) where no model exists — treat a missing
(lab, year) as "no estimate", not zero. SpaceXAI's 2026 row is dated June 30
(not Dec 31) and is net of the Colossus capacity SpaceX sells to Anthropic,
Google, and Reflection AI — its 2024/2025 rows have no such subtraction
because the sale agreements all start May–July 2026.

### Caveats for downstream use

- **Don't derive cross-row totals or ratios from the percentile columns.**
  The lab models share one RNG stream (each reseeds 42), so per-sample draws
  are artificially aligned across labs and years: per-row percentiles are
  valid, but a stacked "all labs" bar with a credible interval — or a
  year-over-year growth CI — is not supported by this table. Stacking the
  medians for display is fine.
- Missing (lab, year) rows mean "no estimate", so a grouped (not stacked)
  layout reads best for the earlier years with partial lab coverage.

### Consolidated page (draft)

```
python3.11 build_compute_page.py
```

regenerates `lab_compute_page_draft.html`, one self-contained page in the
Epoch website style
holding both views of the data: a grouped bar chart of the year-end estimates
(median bars, 90%-CI whiskers, collapsible data table), and one walkthrough
section per (lab × year-end) snapshot showing every intermediate quantity in
model order. The page is a pure view over the two tables from
`get_all_tables()` — new steps or snapshots added to `MODEL_STEPS` appear on
the page (and in the intermediates CSV) with no changes to the viz code (a new
snapshot also needs its (lab, year) → trace-key entry in `LAB_YEAR_KEYS` in
`generate_lab_compute_tables.py`). Re-run it whenever the models or priors change.

## Setup

Python 3.11 with `pip install -r requirements.txt`. Runs need network access
for the Epoch data hub fetch.

Tests:

```
python3.11 -m pytest tests/
```
