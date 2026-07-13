# Data inputs for the lab compute models

Generated exports (do not hand-edit — regenerate with
`python3.11 generate_lab_compute_tables.py` from the repo root):

- `lab_compute_year_end_by_lab.csv` / `lab_compute_intermediates_by_lab.csv` —
  the frontier-lab compute tables (see the root README).

Hand-maintained inputs:

- `IT power by chip.csv` — per-GPU IT power for Nvidia chips (canonical location
  per CLAUDE.md; TPU/Trainium power specs live in `../lab_model_params.csv` as
  `chip_specs` rows). Read by `frontier_lab_compute_model.py` and
  `openai_compute_monte_carlo`.
- `lab IT power.csv` — OpenAI's disclosed IT power by date. Read by the same two.

Chip fleet data (Nvidia per-owner fleets, TPU and AMD cumulative sales) is no
longer stored here — it is fetched at runtime from the Epoch AI data hub via
`../epoch_data.py` (AI Chip Sales and AI Chip Owners datasets), with downloads
cached one-per-day under `../.cache/`.

Legacy snapshot from before the split out of the ai-chip-counts repo:

- `nvidia_calendar_quarter_chip_timelines.csv` — from ai-chip-counts
  `csv_export/`, produced by its `nvidia_estimates.ipynb`. Only read by
  `archive/coreweave_it_power_point_estimate.py`.
