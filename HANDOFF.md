# HANDOFF — current state for agent sessions

Read this before starting work. Before ending a session that changed models,
params, or data: add a dated one-liner under "Recently landed", update or
delete anything that no longer describes reality, and keep this file under
~40 lines. Its git history is the journal — prune freely.

## Source-of-truth map
- Judgment priors + shared chip IT-power specs (`chip_specs` rows):
  `lab_model_params.csv`, read by the lab notebooks and
  `frontier_lab_compute_model.py` (loader in `lab_compute_utils`).
- Model structure: the jupytext-paired notebook `.py` files + the frontier script.
- Chip fleet data (Nvidia owners, TPU/AMD sales): fetched at runtime from the
  Epoch data hub via `epoch_data.py` — no local snapshots.
- `docs/` holds prose artifacts, not sources of truth — trust the sheet and
  code over them.

## Recently landed (prune entries older than ~2 weeks)
- 7/13: repo split out of ai-chip-counts (`ai-lab-compute/`, history preserved
  via git filter-repo). Chip fleet reads rewired from local/root CSVs to
  runtime fetches from the Epoch data hub (`epoch_data.py`; owners data is
  filtered to Nvidia chips to avoid double-counting the TPU/AMD fleets the
  models add separately). Deleted `data/nvidia_owners_cumulative_by_chip.csv`
  and `data/nvidia_cumulative_by_chip.csv`.
- 7/10: lab_compute_tables consolidated page; GDM + Meta end-2024 backcasts
  promoted to canonical; anthropic_2024_backcast + lab_2024_backcasts
  notebooks; MODEL_STEPS intermediate traces exported.
- 7/8: openai watts/GPU = server power × sampled IT overhead; figure_accuracy
  prior added → OpenAI 2025 1.76M. MSL rented-cloud term → 988k. (docs writeup
  NOT updated for the OpenAI changes per Josh — OpenAI section and Meta row
  both stale.)

## Open decisions
- alphabet_activities compute_share CI (0.30–0.70, median 0.46) sits below the
  model's own sizing arithmetic (~0.6): shift up or add justification.
- Correlate the two DeepMind sub-shares (ρ≈0.5) in the headline model, or keep
  independent draws + the sensitivity check?

## Gotchas
- After editing the sheet, re-execute the affected notebook(s) AND the frontier
  script so outputs match; Excel rewrites its last_updated dates (harmless).
- Epoch-hub fetches are cached per day in `.cache/`; delete the cache to force
  a refresh. Published estimates are regenerated periodically, so re-executed
  outputs can drift slightly between days.
- `docs/alphabet-level activities.docx` still says $5.3B for Q1 2026;
  10-Q: $5,391M.
