# Anthropic's compute at the end of 2024

*Generated 2026-07-30 from `notebooks/anthropic_cloud_spend_2024.ipynb` (Part A) in the ai-compute-users repo. Canonical: priors live in `lab_model_params.csv` and the model in `frontier_lab_compute_model.py` (`model_anthropic_2024`), exported to the tables and compute page. Charts live in the notebook.*

**Headline: ~209k H100-equivalents at end-2024 (90% CI 148k–292k)** — reported cloud spending, converted directly into chips at 2024 contract prices.

## The question and the data

How many H100-equivalents (H100e) did Anthropic have at the end of 2024? The one strong anchor — the leaked ~1.4 GW power figure — only describes the end of **2025**. For 2024 the only hard data is reported cloud spend: **$2.5B in 2024** and **$6.8B in 2025** (The Information; sampled as correlated lognormals, ±~12%).

The mainline method converts that spending directly. An experimental alternative — backcasting the end-2025 power-anchored fleet — is kept in the notebook as a validating cross-check only (see below); it is deliberately **not** part of the headline.

## The spending trajectory

Annual totals give the area under the spend curve, not its height at the year boundary. Assuming the spend rate grew exponentially through 2025 and integrates to the 2025 total, the end-2024 rate is pinned by two things: the 2025 total, and the within-2025 growth rate.

The totals ratio fixes the *average* 2024→2025 growth (~2.7×/yr). A **growth-shape factor** scales it: shape 1 is the single smooth exponential through both annual totals; below 1 means front-loaded (2025 started at a high rate); above 1 means back-loaded. Prior: **0.9–1.5** (90% CI).

The narrowed range (rather than a diffuse 0.6–1.6) is motivated by two external reference points:

- **SemiAnalysis's quarterly Anthropic build** (training costs + inference COGS, annualized at quarter midpoints) is calibrated to the same annual totals, so it speaks to the shape of the ramp. It hugs the smooth exponential through 2024, crosses the year boundary right at its rate (~$4B/yr), then runs a little steeper through 2025 — equivalent to shape ≈ 1.3. Front-loaded shapes (≤0.6) need a steeper 2024 ramp than the build shows and finish 2025 too low; heavily back-loaded shapes (≥1.6) imply a nearly flat 2024 that the build contradicts. Their Q1-2026 estimate accelerates sharply to ~$16.0B/yr annualized.
- **WSJ reported** Anthropic's Q1-2026 compute and infrastructure costs at **$3.4B** (~$13.6B/yr annualized) — between the fitted end-2025 rate and SemiAnalysis's steeper Q1-26 build. ([WSJ](https://www.wsj.com/tech/ai/mind-blowing-growth-is-about-to-propel-anthropic-into-its-first-profitable-quarter-7edbf2f4))

Fitted results (5th / median / 95th):

| Quantity | 5th | median | 95th |
|---|---|---|---|
| End-2024 spending rate ($B/yr) | 2.8 | **3.6** | 4.3 |
| End-2025 spending rate ($B/yr) | 9.5 | 11.5 | 14.2 |

## Converting dollars into chips

A rented fleet costing some rate per H100e-hour, billed around the clock, burns that rate × 8,760 hours per year per chip. Divide the end-2024 spending rate by the annual cost of one H100e.

**The 2024 price prior: $1.50–2.50 per H100e-hour** (effective). [SemiAnalysis's GPU pricing index](https://semianalysis.com/gpu-pricing-index/) puts 1-year H100 contracts at ~$3/hr through 2023, falling to the low-$2s through 2024. Anthropic's effective rate should sit below that curve (multi-year terms price below 1-year; scale and pricing power with closely tied cloud providers; the TPU slice entering the mix in 2024) — but can also sit above pure GPU-hour pricing, since "compute spend" plausibly includes auxiliary costs (storage, networking, CPU fleets) beyond GPU-hours. One offsetting pair treated as a wash: long-term contracts don't reprice onto each quarter's cheaper deals (up), but long-term pricing is smoother and lower to begin with (down). Remaining assumption: every chip billed around the clock at the contract rate.

**Result (headline): 209k H100e (90% CI 148k–292k).**

## The experimental cross-check: backcasting the end-2025 fleet

Kept in the notebook, not canonical. It starts from the power model's end-2025 fleet (1.18M H100e, 90% CI 0.85M–1.64M, from the ~1.4 GW leak with a 1.0–1.9 GW band) and shrinks it twice:

> end-2024 fleet = end-2025 fleet × (spending ratio) × (price ratio)

with the spending ratio from the fitted curve (median ~0.31) and a 2025/2024 price ratio built from a 2025 Nvidia blend ($1.30–1.80/H100e-hr, correlated 0.5 with the 2024 price), a Trainium2 bucket (Amazon's ~$0.66/chip-hr × 1.0–1.6 markup, 40–70% of spend), blended harmonically and capped at 0.9.

**Result: 259k H100e (90% CI 148k–436k)** — median ~1.24× the headline.

Why it stays experimental: it chains three multiplicative uncertain factors (fleet × spending ratio × price ratio), so it is wider with a long right tail — its top-5% draws are worlds where a high power-model read, a front-loaded 2025, and a near-capped price ratio all line up — and its extra inputs (the leaked power figure, the 2025 Trainium2 price structure) are judgment-heavier than the 2024 price evidence the headline rests on. As a cross-check it brackets the headline from above rather than contradicting it: the 1.24× gap is equivalent to Anthropic effectively paying ~$1.58/H100e-hr in 2024, inside (if below the median of) the sticker prior. An earlier revision (2026-07-16) mixed the two routes 50/50 for a headline of ~230k; the mixture was dropped on 2026-07-30.

## Assumptions ledger (headline)

| Assumption | Value | Basis |
|---|---|---|
| 2024 cloud spend | $2.5B (2.2–2.85) | The Information |
| 2025 cloud spend | $6.8B (6.0–7.7), corr 0.5 with 2024 | The Information |
| Within-2025 growth shape | 0.9–1.5 | SemiAnalysis quarterly build; WSJ Q1-26 figure |
| 2024 effective price | $1.50–2.50 /H100e-hr | SA 1-yr index; multi-year discounts vs auxiliary costs |
| Billing | around the clock at the contract rate | — |

*Provenance: originally developed as §9 of the standalone cloud-spend notebook, then its own notebook (`anthropic_2024_backcast`), now Part A of `anthropic_cloud_spend_2024` (Part B holds the end-2025 cloud-spend cross-check); `lab_2024_backcasts` and the export tables read the canonical `model_anthropic_2024`.*
