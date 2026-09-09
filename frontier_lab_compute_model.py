"""Consolidated frontier-lab compute Monte Carlo.

One master script holding the model structure and results for five labs'
end-2025 compute, in H100-equivalents (H100e):

    Google DeepMind, Meta Superintelligence Labs (MSL), OpenAI, Anthropic,
    SpaceXAI

The sampled priors live in lab_model_params.csv (loaded via
lab_compute_utils.load_lab_params) — the single source of truth shared with the
lab notebooks. This file holds only model structure. Each lab keeps the
canonical model from its own notebook; the detailed walkthroughs,
visualizations, and sensitivity sweeps live in the individual notebooks
(deepmind_compute_model, msl_compute_model, openai_power_model,
anthropic_power_2025, spacexai_compute_model) and are intentionally
not reproduced here.

For the OpenAI and Anthropic end-2025 estimates the *power-based* model is
canonical; the cloud-spend analyses are deliberately excluded. (Anthropic's
end-2024 is the exception: no power anchor exists for it, so its canonical
model converts cloud spending directly.)

Besides the end-2025 headline models, section 5 holds end-2024 backcasts for
Google DeepMind, Meta AI (pre-MSL), and Anthropic. OpenAI's end-2024 falls out
of model_openai() (its power series is per-year). Section 6 holds the SpaceXAI model, which
produces three snapshots at once (end-2024, end-2025, and mid-2026) from
the Colossus disclosure record as dated by Epoch's data-center timelines; the
mid-2026 snapshot nets out the capacity SpaceX sells to other labs.

Each model also records the intermediate quantities behind its final
distribution in MODEL_STEPS (pure bookkeeping, no effect on results), which
generate_lab_compute_tables.py exports as a table and build_compute_page.py
renders as a walkthrough page.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squigglepy as sq

N_SAMPLES = 5000
H100_FLOPS = 1.979e15  # H100 dense 8-bit FLOP/s, the H100e denominator

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))  # so sibling modules resolve from any cwd
from lab_compute_utils import load_lab_params
from epoch_data import load_nvidia_owners_cumulative, load_chip_sales_cumulative

# Each model reseeds 42 so it reproduces its canonical notebook run. Side
# effect: the labs share one RNG stream, so per-sample values are artificially
# aligned across labs — fine for the per-lab percentiles reported here, but do
# not sum or ratio samples ACROSS labs without reseeding each model differently.


def fmt(x):
    """Format an H100e / chip count as a short string (millions or thousands)."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    return f"{x / 1e3:,.0f}k"


def pctiles(samples):
    """Return (5th, 50th, 95th) percentiles of a sample array."""
    p = sq.get_percentiles(samples, percentiles=[5, 50, 95])
    return p[5], p[50], p[95]


# Each model_*() call refreshes its lab's entry here with an ordered list of
# the intermediate quantities behind its final distribution, so downstream
# exports (generate_lab_compute_tables.py) can show how each estimate is
# built. Pure bookkeeping: recording steps draws no samples and changes no
# results. Read a lab's entry right after calling its model.
MODEL_STEPS = {}


def step(name, label, samples, units, kind, expression=""):
    """One named quantity in a model's decomposition. kind is 'input' (sampled
    prior), 'constant' (fixed scalar), 'derived', or 'final'; samples is a
    sample array (or a scalar for constants); expression says how a derived
    quantity combines earlier steps, referring to them by name."""
    return dict(name=name, label=label, samples=samples, units=units,
                kind=kind, expression=expression)


# ---------------------------------------------------------------------------
# 1. Google DeepMind
# ---------------------------------------------------------------------------
# DeepMind H100e = total_owned x deployment_lag x DeepMind_fraction, where the
# DeepMind fraction blends its share of Google's cloud half and its share of the
# internal (non-cloud) half:
#   fraction = cloud_share * dm_cloud_share + (1 - cloud_share) * dm_noncloud_share
# The two DeepMind shares are drawn with a modest positive correlation (from the
# sheet); the owned fleet = Nvidia GPUs + Google TPUs (drawn independently).

def model_deepmind():
    sq.set_seed(42)
    P = load_lab_params()["deepmind"]
    nvidia_owned = P["nvidia_owned"] @ N_SAMPLES
    google_owned = P["google_owned"] @ N_SAMPLES  # TPU fleet
    total_owned = nvidia_owned + google_owned

    # Only part of the owned fleet is online at any moment.
    deployment_lag = P["deployment_lag"] @ N_SAMPLES
    operational = total_owned * deployment_lag

    # CFO's "around half" cloud vs internal split, and DeepMind's slice of each.
    cloud_share = P["cloud_share"] @ N_SAMPLES
    # The two DeepMind shares plausibly move together, so draw them with a modest
    # positive correlation (from the sheet) rather than independently — independent
    # draws let a high cloud share offset a low non-cloud share and artificially
    # narrow the DeepMind CI. P holds fresh dist objects, safe for sq.correlate to tie.
    dm_cloud_dist, dm_noncloud_dist = sq.correlate(
        (P["dm_cloud_share"], P["dm_noncloud_share"]), P["dm_share_correlation"])
    dm_cloud_share = dm_cloud_dist @ N_SAMPLES      # enterprise Gemini + external rentals
    dm_noncloud_share = dm_noncloud_dist @ N_SAMPLES  # consumer Gemini + DM R&D
    dm_fraction = cloud_share * dm_cloud_share + (1 - cloud_share) * dm_noncloud_share
    dm_h100e = operational * dm_fraction

    MODEL_STEPS["deepmind"] = [
        step("nvidia_owned", "Google-owned Nvidia fleet", nvidia_owned, "H100e", "input"),
        step("google_owned", "Google TPU fleet", google_owned, "H100e", "input"),
        step("total_owned", "Total owned fleet", total_owned, "H100e", "derived",
             "Google-owned Nvidia fleet + Google TPU fleet"),
        step("deployment_lag", "Operational share", deployment_lag, "ratio", "input",
             "share of owned chips that are up and running"),
        step("operational", "Operational fleet", operational, "H100e", "derived",
             "Total owned fleet × Operational share"),
        step("cloud_share", "Cloud share of Google ML compute", cloud_share, "share", "input"),
        step("dm_cloud_share", "DeepMind share of the cloud half", dm_cloud_share, "share", "input"),
        step("dm_noncloud_share", "DeepMind share of the internal half", dm_noncloud_share,
             "share", "input"),
        step("dm_fraction", "DeepMind fraction of the operational fleet", dm_fraction,
             "share", "derived",
             "Cloud share × DeepMind share of the cloud half"
             " + (1 − Cloud share) × DeepMind share of the internal half"),
        step("total_h100e", "DeepMind compute, end-2025", dm_h100e, "H100e", "final",
             "Operational fleet × DeepMind fraction of the operational fleet"),
    ]
    return dm_h100e


# ---------------------------------------------------------------------------
# 2. Meta Superintelligence Labs (MSL)
# ---------------------------------------------------------------------------
# MSL H100e = total_owned x deployment_lag x MSL_share + rented_cloud. Meta's
# fleet is almost entirely internal at end-2025; the MSL share is sampled
# directly (frontier AI work vs core-business recommenders). The rented term
# covers the CoreWeave/Google/Oracle deals signed Sept-Oct 2025: a
# zero-inflated spend run rate converted at GB200/GB300 3-year rental prices
# (SemiAnalysis InferenceX Aug 2025) -- see msl_compute_model section 4.
# Owned fleet = Nvidia GPUs + AMD Instinct (drawn independently); MTIA excluded.

def model_msl():
    sq.set_seed(42)
    P = load_lab_params()["msl"]
    nvidia_owned = P["nvidia_owned"] @ N_SAMPLES
    amd_owned = P["amd_owned"] @ N_SAMPLES
    total_owned = nvidia_owned + amd_owned

    # Slightly higher lag ratios than DeepMind (Meta's fleet ramped fast in 2025).
    deployment_lag = P["deployment_lag"] @ N_SAMPLES
    operational = total_owned * deployment_lag

    # Frontier vs core-business split — highly uncertain, see the sheet's notes.
    msl_share = P["msl_share"] @ N_SAMPLES
    msl_owned = operational * msl_share

    # Rented cloud, no deployment-lag haircut (billed as delivered). Spend run
    # rate ($B/yr) buys H100e at an uncertain price per H100e-hour: the
    # GB200-GB300 3-year rental range ($1.32-1.58), shaded down to $1.20-1.50
    # because Meta's deals run 5-6 years and Meta is a large customer (see the
    # notebook's caveats).
    cloud_spend = sq.zero_inflated(P["cloud_p_nothing_online"],
                                   P["cloud_spend_run_rate"]) @ N_SAMPLES
    price_per_h100e_hour = sq.to(1.20, 1.50) @ N_SAMPLES
    rented_h100e = cloud_spend * 1e9 / (price_per_h100e_hour * 8760)
    msl_h100e = msl_owned + rented_h100e

    MODEL_STEPS["msl"] = [
        step("nvidia_owned", "Meta-owned Nvidia fleet", nvidia_owned, "H100e", "input"),
        step("amd_owned", "Meta-owned AMD Instinct fleet", amd_owned, "H100e", "input"),
        step("total_owned", "Total owned fleet", total_owned, "H100e", "derived",
             "Meta-owned Nvidia fleet + Meta-owned AMD Instinct fleet"),
        step("deployment_lag", "Operational share", deployment_lag, "ratio", "input",
             "share of owned chips that are up and running"),
        step("operational", "Operational fleet", operational, "H100e", "derived",
             "Total owned fleet × Operational share"),
        step("msl_share", "MSL share vs core-business recommenders", msl_share, "share", "input"),
        step("msl_owned", "MSL slice of the owned fleet", msl_owned, "H100e", "derived",
             "Operational fleet × MSL share vs core-business recommenders"),
        step("cloud_spend", "Cloud rental spend run rate", cloud_spend, "USD B/yr", "input"),
        step("rental_price", "Rental price per H100e-hour", price_per_h100e_hour,
             "USD/H100e-hr", "input"),
        step("rented_h100e", "Rented cloud compute", rented_h100e, "H100e", "derived",
             "Cloud rental spend run rate ÷ (Rental price per H100e-hour × 8760 h/yr)"),
        step("total_h100e", "MSL compute, end-2025", msl_h100e, "H100e", "final",
             "MSL slice of the owned fleet + Rented cloud compute"),
    ]
    return msl_h100e


# ---------------------------------------------------------------------------
# 3. OpenAI (power-based model)
# ---------------------------------------------------------------------------
# Turns OpenAI's disclosed IT power per year into H100e via Microsoft's chip
# deployment mix. The fleet is vintage-layered: each year's added power keeps
# the mix Microsoft was deploying then and carries forward -- OpenAI's capacity
# comes through long-term contracts, so it isn't refreshed to newer chips (the
# openai notebook stress-tests this assumption). Five sampled inputs: deployment
# lag (which Microsoft snapshot each year reads), a power definition factor
# (IT vs gross), a figure-accuracy factor (is the internal number itself right),
# rounding jitter, and an IT overhead factor (server power -> IT power per GPU).

QUARTER_DAYS = 365.25 / 4
OAI_CHIP_TYPES = ["A100", "H100/H200", "B200", "B300"]


def _load_openai_data():
    """OpenAI's disclosed power, Microsoft's cumulative fleet, and per-chip specs."""
    owners_df = load_nvidia_owners_cumulative()
    chip_power_df = pd.read_csv(HERE / "data" / "IT power by chip.csv")
    openai_df = pd.read_csv(HERE / "data" / "lab IT power.csv")
    openai_df["Date"] = pd.to_datetime(openai_df["Date"], format="%m/%d/%y")

    # IT watts per GPU. The power CSV names Blackwell "GB200"/"GB300"; the fleet
    # data calls them "B200"/"B300", so translate.
    rename = {"A100": "A100", "H100": "H100/H200", "GB200": "B200", "GB300": "B300"}
    # Median IT watts per GPU (drives the mix shares, which a shared overhead leaves
    # unchanged) and server power per GPU (scaled by the sampled overhead below).
    watts_per_gpu = {
        rename[r["Chip type"]]: r["IT power per GPU (W)"]
        for _, r in chip_power_df.iterrows()
        if r["Chip type"] in rename
    }
    server_power_per_gpu = {
        rename[r["Chip type"]]: r["Server power per GPU (W)"]
        for _, r in chip_power_df.iterrows()
        if r["Chip type"] in rename
    }

    microsoft = owners_df[
        (owners_df["Owner"] == "Microsoft") & (owners_df["Chip type"].isin(OAI_CHIP_TYPES))
    ].copy()
    microsoft["End date"] = pd.to_datetime(microsoft["End date"])
    ms_units = (microsoft.pivot_table(index="End date", columns="Chip type",
                                       values="Number of Units", aggfunc="first")
                .reindex(columns=OAI_CHIP_TYPES).fillna(0).sort_index())
    ms_h100e = (microsoft.pivot_table(index="End date", columns="Chip type",
                                       values="Compute estimate in H100e (median)", aggfunc="first")
                .reindex(columns=OAI_CHIP_TYPES).fillna(0).sort_index())

    dates = list(pd.to_datetime(openai_df["Date"].sort_values().unique()))
    last_date = dates[-1]

    # H100e per GPU is a (roughly constant) hardware ratio; read off the latest snapshot.
    h100e_per_gpu = {
        c: (ms_h100e.loc[last_date, c] / ms_units.loc[last_date, c])
        if ms_units.loc[last_date, c] > 0 else 0.0
        for c in OAI_CHIP_TYPES
    }
    # Microsoft's cumulative IT power (MW) by chip across every quarter.
    ms_power = (ms_units * pd.Series(watts_per_gpu) / 1e6).sort_index()

    # OpenAI's disclosed power per year, and the power added each year.
    disclosed = {d: float(p) for d, p in
                 zip(dates, openai_df.sort_values("Date")["Total IT power (MW)"])}
    added, prev = {}, 0.0
    for d in dates:
        added[d] = disclosed[d] - prev
        prev = disclosed[d]

    return dict(dates=dates, last_date=last_date, watts_per_gpu=watts_per_gpu,
                server_power_per_gpu=server_power_per_gpu,
                h100e_per_gpu=h100e_per_gpu, ms_power=ms_power,
                disclosed=disclosed, added=added)


def _chip_power_shares(data, lag_quarters):
    """For each OpenAI year, the share of power on each chip under a deployment lag,
    as a {date: {chip: share}} dict (vintage-layered: each year's additions keep
    their deployment mix and carry forward). lag_quarters may be a scalar or a
    per-sample array (shares come back the same shape)."""
    dates, ms_power = data["dates"], data["ms_power"]
    added_power = data["added"]
    start = ms_power.index[0]
    ms_day = np.array([(d - start).days for d in ms_power.index], dtype=float)
    lag_days = lag_quarters * QUARTER_DAYS

    # Microsoft's cumulative power by chip as seen `lag` quarters before each date,
    # interpolated to the (possibly fractional) day.
    cumulative = {
        d: {c: np.interp((d - start).days - lag_days, ms_day, ms_power[c].values)
            for c in OAI_CHIP_TYPES}
        for d in dates
    }
    # Mix of power Microsoft *added* each step (cumulative for the first step so
    # legacy A100s count; incremental afterward).
    added_mix = {}
    for i, d in enumerate(dates):
        if i == 0:
            add = {c: np.maximum(cumulative[d][c], 0.0) for c in OAI_CHIP_TYPES}
        else:
            prev = dates[i - 1]
            add = {c: np.maximum(cumulative[d][c] - cumulative[prev][c], 0.0)
                   for c in OAI_CHIP_TYPES}
        tot = sum(add.values())
        added_mix[d] = {c: add[c] / tot for c in OAI_CHIP_TYPES}

    # Carry each year's additions forward at their own mix.
    shares = {}
    carried = {c: 0.0 for c in OAI_CHIP_TYPES}
    for d in dates:
        for c in OAI_CHIP_TYPES:
            carried[c] = carried[c] + added_power[d] * added_mix[d][c]
        carried_tot = sum(carried.values())
        shares[d] = {c: carried[c] / carried_tot for c in OAI_CHIP_TYPES}
    return shares


def model_openai():
    """Returns the end-2025 total H100e samples (plus per-year totals for every
    disclosed year-end) and the per-chip specs and chip counts that the
    Anthropic model borrows."""
    sq.set_seed(42)
    data = _load_openai_data()
    watts, h100e_per_gpu = data["watts_per_gpu"], data["h100e_per_gpu"]
    server_ppg = data["server_power_per_gpu"]
    dates, last_date = data["dates"], data["last_date"]

    # Sampled inputs. The lag shifts which Microsoft snapshot each year reads.
    _params = load_lab_params()
    P = _params["openai"]
    lag_quarters = P["lag_quarters"] @ N_SAMPLES
    # Watts per GPU = server power x a shared IT overhead factor (server -> IT power);
    # a higher overhead means fewer chips per disclosed MW. Its low end is the upside.
    it_overhead = _params["chip_specs"]["nvidia_it_overhead"] @ N_SAMPLES

    # Total power per year: one shared power-definition factor times each
    # disclosure, plus independent rounding jitter. Is each disclosed figure IT
    # power or gross? With probability p_gross it's gross and gets divided by a
    # datacenter PUE (a downward haircut); otherwise it's already IT.
    if_gross_power = 1 / P["gross_pue"]
    definition_factor = sq.mixture([P["if_it_power"], if_gross_power],
                                   [1 - P["p_gross"], P["p_gross"]]) @ N_SAMPLES
    # Is OpenAI's internal figure itself right, aside from rounding and IT-vs-gross?
    # Undercounted providers push up, overstatement down (median ~1.04).
    accuracy_factor = P["figure_accuracy"] @ N_SAMPLES
    # Rounding jitter: the disclosures' 0.1 GW rounding half-step, triangular
    # (edges less likely). Drawn independently per year.
    rounding_mw = P["rounding_mw"]
    rounding_jitter = {
        d: sq.triangular(-rounding_mw, 0.0, rounding_mw) @ N_SAMPLES for d in dates
    }
    total_power = {
        d: (data["disclosed"][d] + rounding_jitter[d])
        * definition_factor * accuracy_factor
        for d in dates
    }

    shares = _chip_power_shares(data, lag_quarters)
    # Chip counts at every disclosed year-end, not just the latest: each year's
    # mix is sized by that year's power.
    counts_by_date = {}
    for d in dates:
        counts_by_date[d] = {}
        for c in OAI_CHIP_TYPES:
            megawatts = total_power[d] * shares[d][c]
            counts_by_date[d][c] = megawatts * 1e6 / (server_ppg[c] * it_overhead)
    counts = counts_by_date[last_date]
    total_h100e_by_date = {
        d: sum(counts_by_date[d][c] * h100e_per_gpu[c] for c in OAI_CHIP_TYPES)
        for d in dates
    }

    MODEL_STEPS["openai"] = [
        step("disclosed_power", "Disclosed end-2025 power", data["disclosed"][last_date],
             "MW", "constant"),
        step("rounding_jitter", "Rounding adjustment (±)", rounding_jitter[last_date],
             "MW", "input",
             "the disclosed figure is rounded to 0.1 GW, so the true value sits"
             " within half a step either way"),
        step("definition_factor", "Power-definition factor (IT vs gross)", definition_factor,
             "ratio", "input",
             "blend of two cases — figure is already IT power (≈1),"
             " or it's gross power ÷ data-center PUE"),
        step("accuracy_factor", "Figure-accuracy factor", accuracy_factor, "ratio", "input"),
        step("total_power", "Modelled end-2025 IT power", total_power[last_date],
             "MW", "derived",
             "(Disclosed end-2025 power + Rounding adjustment)"
             " × Power-definition factor × Figure-accuracy factor"),
        step("lag_quarters", "Deployment lag behind Nvidia's sales mix", lag_quarters,
             "quarters", "input"),
        step("it_overhead", "Server-to-IT power overhead", it_overhead, "ratio", "input"),
        step("total_h100e", "OpenAI compute, end-2025", total_h100e_by_date[last_date],
             "H100e", "final",
             "Modelled end-2025 IT power spread across Nvidia's chip sales mix"
             " (shifted by the Deployment lag), each chip type's power turned into"
             " chips at its server watts × Server-to-IT power overhead, then summed"
             " as H100-equivalents"),
    ]

    return dict(total_h100e=total_h100e_by_date[last_date],
                total_h100e_by_date=total_h100e_by_date, watts_per_gpu=watts,
                h100e_per_gpu=h100e_per_gpu, counts=counts, last_date=last_date)


# ---------------------------------------------------------------------------
# 4. Anthropic (power-based model)
# ---------------------------------------------------------------------------
# Anthropic H100e = total_power_mw x blended_H100e_per_mw, where the blend is set
# by the Trainium2 share of power. At a fixed power budget the Nvidia mix and the
# TPU mix buy about the same H100e per watt, while Trainium2 buys ~0.75x as much
# (per the New Carlisle equivalency in the sheet's chip_specs rows); so the fleet
# collapses to two buckets and the Trainium2 power share is the lever.
# Nvidia specs and the H100:Blackwell ratio are borrowed from the OpenAI model as
# point values; sampled multipliers restore each bucket's efficiency uncertainty
# (nontrainium_eff for Nvidia+TPU, trainium_eff for Trainium2's estimated
# TDP x Nvidia-derived overhead).

# Shared hardware constants from the params sheet's chip_specs rows: TPU TDPs,
# the IT-power overhead, and the supplied Trainium2 equivalency (a fleet worth a
# known H100e draws a known IT power, which fixes the watts per chip).
CHIP_SPECS = load_lab_params()["chip_specs"]

TRAINIUM2_H100E = 1299 / 1979  # Trainium2 dense 8-bit throughput relative to an H100
TRAINIUM2_IT_WATTS = TRAINIUM2_H100E / (
    CHIP_SPECS["trainium2_ref_h100e"] / CHIP_SPECS["trainium2_ref_it_mw"]) * 1e6
IT_OVERHEAD = CHIP_SPECS["tpu_it_overhead"]  # IT power per chip / TDP, for TPUs (no public server specs)


def _tpu_mix_per_mw():
    """Google's real v5+ TPU fleet efficiency (H100e per MW), scored on native
    8-bit peak, weighted by chip count x IT power -- the OpenAI methodology."""
    tpu_tdp_w = {
        "TPU v5e": CHIP_SPECS["tpu_v5e_tdp"],
        "TPU v5p": CHIP_SPECS["tpu_v5p_tdp"],
        "TPU v6e": CHIP_SPECS["tpu_v6e_tdp"],
        "TPU v7": CHIP_SPECS["tpu_v7_tdp"],
    }
    tpu_8bit = {"TPU v5e": 3.93e14, "TPU v5p": 9.18e14, "TPU v6e": 1.836e15, "TPU v7": 4.614e15}
    it_watts = {c: tpu_tdp_w[c] * IT_OVERHEAD for c in tpu_tdp_w}
    h100e_per_chip = {c: tpu_8bit[c] / H100_FLOPS for c in tpu_tdp_w}

    df = load_chip_sales_cumulative("Google")
    snap = df[df["End date"] == pd.Timestamp("2025-12-31")]
    units = {c: float(snap.loc[snap["Chip type"] == c, "Number of units (median)"].iloc[0])
             for c in tpu_tdp_w}
    total_it_mw = sum(units[c] * it_watts[c] / 1e6 for c in tpu_tdp_w)
    return sum(units[c] * h100e_per_chip[c] for c in tpu_tdp_w) / total_it_mw


def model_anthropic(openai_result):
    sq.set_seed(42)
    watts = openai_result["watts_per_gpu"]
    h100e_per_gpu = openai_result["h100e_per_gpu"]
    counts = openai_result["counts"]

    # OpenAI's end-2025 Hopper:Blackwell count ratio (A100 dropped, GB300 folded in).
    hopper = float(np.median(counts["H100/H200"]))
    blackwell = float(np.median(counts["B200"] + counts["B300"]))
    h100_count_fraction = hopper / (hopper + blackwell)

    # Per-chip H100e per MW for the modelled chips.
    h100e_per_mw = {
        "H100": h100e_per_gpu["H100/H200"] / watts["H100/H200"] * 1e6,
        "GB200": h100e_per_gpu["B200"] / watts["B200"] * 1e6,
        "Trainium2": TRAINIUM2_H100E / TRAINIUM2_IT_WATTS * 1e6,
    }
    # Nvidia mix: H100 + GB200 in OpenAI's count ratio, weighted by watts per chip.
    h100_per_gb200 = h100_count_fraction / (1 - h100_count_fraction)
    h100_power_share = (h100_per_gb200 * watts["H100/H200"]) / (
        h100_per_gb200 * watts["H100/H200"] + watts["B200"])
    nvidia_mix_per_mw = (h100_power_share * h100e_per_mw["H100"]
                         + (1 - h100_power_share) * h100e_per_mw["GB200"])

    # Nvidia and TPU are within a few percent, so treat non-Trainium as one bucket.
    nontrainium_per_mw = (nvidia_mix_per_mw + _tpu_mix_per_mw()) / 2
    trainium2_per_mw = h100e_per_mw["Trainium2"]

    # Total power: OpenAI-memo mainline 1.4 GW; 90% CI 1.0-1.9 GW (geomean
    # ~1.38). The top end matches OpenAI's 1.9 GW point figure — allowed since
    # OpenAI's own power is uncertain, so that world implies a larger OpenAI.
    P = load_lab_params()["anthropic"]
    power_mw = (P["lab_power_gw"] @ N_SAMPLES) * 1000.0

    # Trainium2 share of IT power: ~normal, median ~0.52, anchored on New Carlisle
    # + Madison site power (Epoch's data-center directory) and ceiling-checked
    # against Amazon's ~1.4M deployed chips. Sampled independently of power.
    trainium_share = P["trainium_share"] @ N_SAMPLES

    # The non-Trainium bucket's specs above are borrowed point values; this
    # multiplier restores the fleet-efficiency uncertainty the OpenAI model
    # samples (lag + IT overhead), widened a bit for the TPU side.
    nontrainium_eff = P["nontrainium_eff"] @ N_SAMPLES

    # Trainium2's ~870 W per chip is an estimated TDP times an overhead borrowed
    # from Nvidia hardware; this multiplier collapses both uncertainties into one.
    trainium_eff = P["trainium_eff"] @ N_SAMPLES

    blended_per_mw = (trainium_share * trainium2_per_mw * trainium_eff
                      + (1 - trainium_share) * nontrainium_per_mw * nontrainium_eff)
    anthropic_h100e = power_mw * blended_per_mw

    MODEL_STEPS["anthropic"] = [
        step("power_mw", "Total IT power", power_mw, "MW", "input",
             "Leaked lab power (GW) × 1000"),
        step("trainium_share", "Trainium2 share of IT power", trainium_share, "share", "input"),
        step("trainium2_per_mw", "Trainium2 fleet efficiency", trainium2_per_mw,
             "H100e/MW", "constant"),
        step("trainium_eff", "Trainium2 efficiency — uncertainty adjustment", trainium_eff,
             "ratio", "input"),
        step("nontrainium_per_mw", "Nvidia + TPU fleet efficiency", nontrainium_per_mw,
             "H100e/MW", "constant"),
        step("nontrainium_eff", "Nvidia + TPU efficiency — uncertainty adjustment", nontrainium_eff,
             "ratio", "input"),
        step("blended_per_mw", "Blended fleet efficiency", blended_per_mw, "H100e/MW", "derived",
             "Trainium2 share of IT power × Trainium2 fleet efficiency"
             " × Trainium2 efficiency adjustment"
             " + (1 − Trainium2 share of IT power) × Nvidia + TPU fleet efficiency"
             " × Nvidia + TPU efficiency adjustment"),
        step("total_h100e", "Anthropic compute, end-2025", anthropic_h100e, "H100e", "final",
             "Total IT power × Blended fleet efficiency"),
    ]
    return anthropic_h100e


# ---------------------------------------------------------------------------
# 5. End-2024 backcasts: Google DeepMind, Meta AI (pre-MSL), and Anthropic
# ---------------------------------------------------------------------------
# DeepMind and Meta share the top-down shape of the end-2025 models (owned
# fleet x operational ratio x lab share), promoted from the lab_2024_backcasts
# notebook, with two changes in how the first two factors are obtained:
#
#  - Owned fleets are read from Epoch's published quarterly estimates at
#    end-2024, as lognormals through the summed per-chip 5th/95th columns.
#    (Summing per-chip percentile bounds treats chips as perfectly correlated,
#    so the CIs are on the generous side -- the convention the end-2025 sheet
#    rows effectively used.)
#  - The operational/owned ratio is computed from the owned-stock trajectory
#    under a sampled deployment lag instead of hand-derived: fleets grew
#    ~3.5-4.5x during 2024, so the 2025 ratios would overstate early years.
#
# "The lab" in 2024 means frontier-AI compute at the company: MSL did not exist
# (its predecessor was Meta AI / GenAI plus FAIR), and the share priors are for
# those predecessor scopes.

END_2024 = pd.Timestamp("2024-12-31")


def owner_quarterly_h100e_medians(df, owner=None):
    """Quarterly cumulative H100e medians (summed across chip types) as a
    Series indexed by quarter-end date; optionally one owner's slice."""
    if owner is not None:
        df = df[df["Owner"] == owner]
    return df.groupby("End date")["Compute estimate in H100e (median)"].sum()


def end_2024_fleet_dist(df, owner=None):
    """Owned-fleet H100e at end-2024, as a lognormal through the summed
    per-chip 5th/95th columns. The owners data and the chip-sales data name
    those columns differently."""
    if owner is not None:
        df = df[df["Owner"] == owner]
        lo_col, hi_col = "H100e (5th percentile)", "H100e (95th percentile)"
    else:
        lo_col, hi_col = ("Compute estimate in H100e (5th percentile)",
                          "Compute estimate in H100e (95th percentile)")
    snap = df[df["End date"] == END_2024]
    return sq.to(snap[lo_col].sum(), snap[hi_col].sum())


def operational_ratio_2024(stock_series, lag_quarters):
    """Owned stock `lag_quarters` before end-2024, as a fraction of the
    end-2024 stock, interpolated along the quarterly median trajectory. The
    trajectory's shape is treated as data; the stock's level uncertainty is
    sampled separately (the fleet lognormals)."""
    window = stock_series.loc["2023-12-31":END_2024]
    days = np.array([(d - window.index[0]).days for d in window.index], dtype=float)
    target = days[-1] - lag_quarters * QUARTER_DAYS
    return np.interp(target, days, window.values) / window.values[-1]


def model_deepmind_2024():
    sq.set_seed(42)
    P = load_lab_params()["deepmind"]
    nvidia_owned = end_2024_fleet_dist(load_nvidia_owners_cumulative(), "Google") @ N_SAMPLES
    google_owned = end_2024_fleet_dist(load_chip_sales_cumulative("Google")) @ N_SAMPLES  # TPUs
    total_owned = nvidia_owned + google_owned

    # Operational share of owned: sample the install lag, then read the owned
    # stock that many quarters before end-2024 off the trajectory.
    lag_quarters = P["lag_quarters_2024"] @ N_SAMPLES
    stock = (owner_quarterly_h100e_medians(load_nvidia_owners_cumulative(), "Google")
             + owner_quarterly_h100e_medians(load_chip_sales_cumulative("Google"))).dropna()
    deployment_lag = operational_ratio_2024(stock, lag_quarters)
    operational = total_owned * deployment_lag

    # One overall DeepMind share: Google gave no cloud/internal split for 2024,
    # so the 2025 model's two-sub-share blend has nothing to anchor on.
    dm_share = P["dm_share_2024"] @ N_SAMPLES
    dm_h100e = operational * dm_share

    MODEL_STEPS["deepmind_2024"] = [
        step("nvidia_owned", "Google-owned Nvidia fleet", nvidia_owned, "H100e", "input"),
        step("google_owned", "Google TPU fleet", google_owned, "H100e", "input"),
        step("total_owned", "Total owned fleet", total_owned, "H100e", "derived",
             "Google-owned Nvidia fleet + Google TPU fleet"),
        step("lag_quarters_2024", "Deployment lag", lag_quarters, "quarters", "input",
             "delay from owning chips to running them"),
        step("deployment_lag", "Operational share", deployment_lag, "ratio", "derived",
             "owned stock one Deployment lag earlier ÷ end-2024 stock"),
        step("operational", "Operational fleet", operational, "H100e", "derived",
             "Total owned fleet × Operational share"),
        step("dm_share_2024", "DeepMind share of Google ML compute", dm_share, "share", "input"),
        step("total_h100e", "DeepMind compute, end-2024", dm_h100e, "H100e", "final",
             "Operational fleet × DeepMind share of Google ML compute"),
    ]
    return dm_h100e


def model_msl_2024():
    """Meta AI / GenAI (the pre-MSL frontier org) at end-2024. Owned fleet =
    Meta's Nvidia GPUs plus a sampled slice of the all-owner AMD Instinct
    fleet (the dashboards don't split AMD by owner); MTIA excluded."""
    sq.set_seed(42)
    P = load_lab_params()["msl"]
    nvidia_owned = end_2024_fleet_dist(load_nvidia_owners_cumulative(), "Meta") @ N_SAMPLES
    amd_all_owners = end_2024_fleet_dist(load_chip_sales_cumulative("AMD")) @ N_SAMPLES
    meta_amd_share = P["meta_amd_share_2024"] @ N_SAMPLES
    amd_owned = amd_all_owners * meta_amd_share
    total_owned = nvidia_owned + amd_owned

    # Operational share of owned, as in the DeepMind backcast. The trajectory
    # uses the median AMD share; only the level uncertainty is sampled.
    lag_quarters = P["lag_quarters_2024"] @ N_SAMPLES
    nvidia_stock = owner_quarterly_h100e_medians(load_nvidia_owners_cumulative(), "Meta")
    amd_stock = (owner_quarterly_h100e_medians(load_chip_sales_cumulative("AMD"))
                 .reindex(nvidia_stock.index).fillna(0.0))
    stock = (nvidia_stock + float(np.median(meta_amd_share)) * amd_stock).dropna()
    deployment_lag = operational_ratio_2024(stock, lag_quarters)
    operational = total_owned * deployment_lag

    # Frontier vs core-business split for the predecessor org, see the sheet.
    meta_ai_share = P["meta_ai_share_2024"] @ N_SAMPLES
    meta_h100e = operational * meta_ai_share

    MODEL_STEPS["msl_2024"] = [
        step("nvidia_owned", "Meta-owned Nvidia fleet", nvidia_owned, "H100e", "input"),
        step("amd_all_owners", "AMD Instinct fleet, all owners", amd_all_owners,
             "H100e", "input"),
        step("meta_amd_share_2024", "Meta share of the AMD fleet", meta_amd_share,
             "share", "input"),
        step("amd_owned", "Meta-owned AMD Instinct fleet", amd_owned, "H100e", "derived",
             "AMD Instinct fleet, all owners × Meta share of the AMD fleet"),
        step("total_owned", "Total owned fleet", total_owned, "H100e", "derived",
             "Meta-owned Nvidia fleet + Meta-owned AMD Instinct fleet"),
        step("lag_quarters_2024", "Deployment lag", lag_quarters, "quarters", "input",
             "delay from owning chips to running them"),
        step("deployment_lag", "Operational share", deployment_lag, "ratio", "derived",
             "owned stock one Deployment lag earlier ÷ end-2024 stock"),
        step("operational", "Operational fleet", operational, "H100e", "derived",
             "Total owned fleet × Operational share"),
        step("meta_ai_share_2024", "Meta AI (pre-MSL) frontier share", meta_ai_share,
             "share", "input"),
        step("total_h100e", "Meta AI frontier compute, end-2024", meta_h100e, "H100e", "final",
             "Operational fleet × Meta AI (pre-MSL) frontier share"),
    ]
    return meta_h100e


def model_anthropic_2024():
    """Anthropic at end-2024, converted directly from cloud spending. No power
    anchor exists for 2024 (the leaked ~1.4 GW describes end-2025), so the
    estimate turns the end-2024 *rate* of cloud spending into chips at 2024
    contract prices: the reported annual spend totals pin an exponential spend
    curve, whose height at the 2024/2025 boundary is divided by the annual
    cost of one H100e billed around the clock. The walkthrough (and an
    experimental power-model backcast deliberately NOT used here) lives in
    notebooks/anthropic_cloud_spend_2024, which samples in the same order as this
    function so the two match exactly under the shared seed."""
    sq.set_seed(42)
    P = load_lab_params()["anthropic"]

    # Reported full-year cloud spend totals, correlated (same reporting).
    spend24_dist, spend25_dist = sq.correlate(
        (P["cloud_spend_2024"], P["cloud_spend_2025"]), 0.5)
    spend_2024 = (spend24_dist @ N_SAMPLES) * 1e9
    spend_2025 = (spend25_dist @ N_SAMPLES) * 1e9

    # The annual totals fix the average 2024->2025 growth; the shape factor
    # says when within 2025 the ramp happened. The exponential that integrates
    # to the 2025 total then gives the spending rate at the year boundary
    # (which is both where 2025 starts and where 2024 ends).
    growth_shape = P["spend_growth_shape_2025"] @ N_SAMPLES
    inst_rate = np.log(spend_2025 / spend_2024) * growth_shape
    runrate_2024_end = (spend_2025 * inst_rate / (1 - np.exp(-inst_rate))
                        * np.exp(-inst_rate))

    # A rented chip billed around the clock costs its hourly rate for each of
    # the year's 8,760 hours.
    price_2024 = P["effective_price_2024"] @ N_SAMPLES
    anthropic_2024_h100e = runrate_2024_end / (price_2024 * 8760)

    MODEL_STEPS["anthropic_2024"] = [
        step("cloud_spend_2024", "2024 cloud spend (full year)", spend_2024 / 1e9,
             "USD B/yr", "input"),
        step("cloud_spend_2025", "2025 cloud spend (full year)", spend_2025 / 1e9,
             "USD B/yr", "input"),
        step("spend_growth_shape_2025", "Within-2025 growth shape", growth_shape,
             "multiplier", "input",
             "multiplier on the average 2024→2025 spend-growth rate: 1 = steady"
             " exponential growth; above 1 = growth concentrated later in 2025,"
             " so spending was still low at the year boundary"),
        step("runrate_2024_end", "End-2024 spending rate", runrate_2024_end / 1e9,
             "USD B/yr", "derived",
             "the exponential spend curve through both annual totals, read at the year boundary"),
        step("effective_price_2024", "2024 effective price", price_2024,
             "USD/H100e-hr", "input"),
        step("total_h100e", "Anthropic compute, end-2024", anthropic_2024_h100e,
             "H100e", "final",
             "End-2024 spending rate ÷ (2024 effective price × 8760 h/yr)"),
    ]
    return anthropic_2024_h100e


# ---------------------------------------------------------------------------
# 6. SpaceXAI (Colossus-anchored; end-2024, end-2025, and mid-2026)
# ---------------------------------------------------------------------------
# SpaceXAI H100e = Colossus capacity + other compute - cloud sales, per the
# methodology of https://epoch.ai/gradient-updates/frontier-labs-dont-use-most-ai-compute:
# nearly all of SpaceXAI's compute is the two Memphis-area Colossus campuses,
# so the fleet is anchored directly on the Colossus disclosure record (S-1 chip
# counts, company announcements) as dated by Epoch's data-center timelines,
# rather than on chip-fleet accounting. "Other compute" is a small buffer for
# capacity outside the two campuses: third-party cloud purchases (bounded by
# the S-1's blended infrastructure-and-cloud expense lines, roughly $1-2B/yr
# at end-2025) plus minor owned sites (Atlanta, Portland).
#
# Colossus at each snapshot is a firm disclosed level plus the in-progress
# phase times how much of it is done. The component sizes are literals taken
# from the disclosure record (Epoch's C2 levels are the S-1 chip counts x
# 2.527 H100e per GB200/GB300); the spacexai_compute_model notebook derives
# them in prose and asserts an exact match against this function.
#
#   end-2024: Colossus 1 phase 1 (100k H100s, online Sept 2024) + phase 2 (the
#             second 100k Hoppers, credited by Epoch in Feb 2025) x a sampled
#             completion share.
#   end-2025: C1 complete (276k) + C2 cluster 1 (278k, S-1) as a disclosure-
#             pinned floor + C2 cluster 2 (278k, Epoch-dated Apr 6 2026) x a
#             completion share that is 0 with probability 1 - p (nothing
#             energized yet, the central case) and a clipped-normal share-if-
#             live draw otherwise.
#   mid-2026: everything through C2's 400+ MW phase 3 sampled as one firm
#             level (Epoch's 1.39M +/-10%) + the pending ~Feb 2027 phase x a
#             sampled fraction open.
#
# The mid-2026 snapshot nets out the capacity SpaceX *sells* to other labs —
# Anthropic (all of Colossus 1 since May 2026, possibly spilling into C2),
# Google (one ~110k-GPU C2 cluster, ramping toward Sept 2026), and Reflection
# AI (small C2 carve-out from July 1) — because the estimate targets
# SpaceXAI's own AI effort, including Cursor (an internal allocation per the
# S-1), not the SpaceX total. Sales subtract the same point levels that went
# into the fleet, so C1 measurement error nets out. No subtractions before
# 2026: the sale agreements all start May-July 2026.

# Colossus component sizes (H100e), from the disclosure record. Keep matched
# with the literals in notebooks/spacexai_compute_model.py.
COLOSSUS_1_PHASE1 = 100_000        # 100k H100s online by Sept 2024 (announcement)
COLOSSUS_1_PHASE2 = 100_000        # second 100k Hoppers, credited by Epoch Feb 2025
COLOSSUS_1 = 276_000               # 150k H100 + 50k H200 + 30k B200 (complete)
COLOSSUS_2_CLUSTER1 = 278_000      # S-1: ~110k GB200, online Oct 19 2025
COLOSSUS_2_CLUSTER2 = 278_000      # S-1: 110k GB300, Epoch-dated Apr 6 2026
COLOSSUS_2_PENDING_FEB2027 = 712_000   # Epoch's projected ~Feb 2027 expansion


def model_spacexai():
    """Returns {"2024": ..., "2025": ..., "h1_2026": ...} H100e sample arrays.
    The 2024/2025 snapshots are the whole fleet; mid-2026 is net of sales.
    Samples the sheet priors in the same order as the notebook under the same
    seed, so the two match exactly."""
    sq.set_seed(42)
    P = load_lab_params()["spacexai"]
    results = {}

    # --- End-2024: Colossus 1 mid-ramp ------------------------------------
    c1_phase2_complete = P["c1_phase2_complete_2024"] @ N_SAMPLES
    colossus_2024 = COLOSSUS_1_PHASE1 + COLOSSUS_1_PHASE2 * c1_phase2_complete
    other_2024 = P["other_compute_2024"] @ N_SAMPLES
    results["2024"] = colossus_2024 + other_2024
    MODEL_STEPS["spacexai_2024"] = [
        step("c1_phase1", "Colossus 1 phase 1 (100k H100s, online Sept 2024)",
             COLOSSUS_1_PHASE1, "H100e", "constant"),
        step("c1_phase2", "Colossus 1 phase 2 (second 100k Hoppers)",
             COLOSSUS_1_PHASE2, "H100e", "constant"),
        step("c1_phase2_complete", "Share of phase 2 online at Dec 31",
             c1_phase2_complete, "share", "input"),
        step("colossus", "Colossus operational capacity", colossus_2024, "H100e", "derived",
             "Colossus 1 phase 1 + Colossus 1 phase 2 × Share of phase 2 online at Dec 31"),
        step("other_compute", "Other sites + cloud purchases", other_2024, "H100e", "input"),
        step("total_h100e", "SpaceXAI compute, end-2024", results["2024"], "H100e", "final",
             "Colossus operational capacity + Other sites + cloud purchases"),
    ]

    # --- End-2025: disclosure-pinned floor plus a zero-heavy share of cluster 2
    p_any_live = P["c2_cluster2_any_live_2025"]            # const: plain float
    share_if_live = P["c2_cluster2_share_if_live_2025"]    # clipped normal
    c2_cluster2_complete = sq.mixture([sq.const(0), share_if_live],
                                      [1 - p_any_live, p_any_live]) @ N_SAMPLES
    colossus_floor_2025 = COLOSSUS_1 + COLOSSUS_2_CLUSTER1
    colossus_2025 = colossus_floor_2025 + COLOSSUS_2_CLUSTER2 * c2_cluster2_complete
    other_2025 = P["other_compute_2025"] @ N_SAMPLES
    results["2025"] = colossus_2025 + other_2025
    MODEL_STEPS["spacexai_2025"] = [
        step("colossus_floor", "Colossus 1 complete + Colossus 2 cluster 1 (S-1 floor)",
             colossus_floor_2025, "H100e", "constant"),
        step("c2_cluster2", "Colossus 2 cluster 2 (110k GB300, S-1)",
             COLOSSUS_2_CLUSTER2, "H100e", "constant"),
        step("c2_any_live", "Probability any of cluster 2 was live at Dec 31",
             p_any_live, "probability", "constant"),
        step("c2_cluster2_complete", "Share of cluster 2 online at Dec 31",
             c2_cluster2_complete, "share", "input"),
        step("colossus", "Colossus operational capacity", colossus_2025, "H100e", "derived",
             "Colossus floor + Colossus 2 cluster 2 × Share of cluster 2 online at Dec 31"),
        step("other_compute", "Other sites + cloud purchases", other_2025, "H100e", "input"),
        step("total_h100e", "SpaceXAI compute, end-2025", results["2025"], "H100e", "final",
             "Colossus operational capacity + Other sites + cloud purchases"),
    ]

    # --- Mid-2026: the fleet, less capacity sold to other labs ----------------
    colossus_firm = P["colossus_firm_h1_2026"] @ N_SAMPLES
    phase_open = P["c2_next_phase_open_h1_2026"] @ N_SAMPLES
    colossus_h1_2026 = colossus_firm + COLOSSUS_2_PENDING_FEB2027 * phase_open
    other_h1_2026 = P["other_compute_h1_2026"] @ N_SAMPLES
    fleet = colossus_h1_2026 + other_h1_2026

    anthropic_spillover = P["anthropic_c2_spillover"] @ N_SAMPLES
    anthropic_sold = COLOSSUS_1 + anthropic_spillover
    google_ramp = P["google_ramp_share"] @ N_SAMPLES
    google_sold = COLOSSUS_2_CLUSTER1 * google_ramp
    reflection_sold = P["reflection_sold_h100e"] @ N_SAMPLES
    internal = fleet - anthropic_sold - google_sold - reflection_sold
    results["h1_2026"] = internal
    MODEL_STEPS["spacexai_h1_2026"] = [
        step("colossus_firm", "Colossus firm capacity (through C2 phase 3)",
             colossus_firm, "H100e", "input"),
        step("c2_pending", "Colossus 2, next projected phase (~Feb 2027)",
             COLOSSUS_2_PENDING_FEB2027, "H100e", "constant"),
        step("c2_phase_open", "Share of the next phase open", phase_open, "share", "input"),
        step("colossus", "Colossus operational capacity", colossus_h1_2026, "H100e", "derived",
             "Colossus firm capacity + Colossus 2 next projected phase × Share of the next phase open"),
        step("other_compute", "Other sites + cloud purchases", other_h1_2026, "H100e", "input"),
        step("fleet", "Total SpaceX fleet", fleet, "H100e", "derived",
             "Colossus operational capacity + Other sites + cloud purchases"),
        step("c1_level", "Colossus 1 (sold to Anthropic)", COLOSSUS_1, "H100e", "constant"),
        step("anthropic_spillover", "Anthropic spillover into C2",
             anthropic_spillover, "H100e", "input"),
        step("anthropic_sold", "Sold to Anthropic (all of C1 + spillover)", anthropic_sold,
             "H100e", "derived", "Colossus 1 + Anthropic spillover into C2"),
        step("c2_cluster1", "Colossus 2 cluster 1 (the Google deal)",
             COLOSSUS_2_CLUSTER1, "H100e", "constant"),
        step("google_ramp_share", "Google ramp share by June 30", google_ramp, "share", "input"),
        step("google_sold", "Sold to Google (one C2 cluster, ramping)", google_sold,
             "H100e", "derived", "Colossus 2 cluster 1 × Google ramp share by June 30"),
        step("reflection_sold", "Sold to Reflection AI", reflection_sold, "H100e", "input"),
        step("total_h100e", "SpaceXAI compute, mid-2026", internal, "H100e", "final",
             "Total SpaceX fleet − Sold to Anthropic − Sold to Google − Sold to Reflection AI"),
    ]
    return results


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def main():
    openai_result = model_openai()
    spacexai = model_spacexai()
    labs = {
        "Google DeepMind": model_deepmind(),
        "Meta SL": model_msl(),
        "OpenAI": openai_result["total_h100e"],
        "Anthropic": model_anthropic(openai_result),
        "SpaceXAI": spacexai["2025"],
    }

    print("End-2025 compute, H100-equivalents (5th / median / 95th):\n")
    print(f"  {'Lab':<18}{'5th':>10}{'median':>10}{'95th':>10}")
    for name, samples in labs.items():
        lo, mid, hi = pctiles(samples)
        print(f"  {name:<18}{fmt(lo):>10}{fmt(mid):>10}{fmt(hi):>10}")

    openai_2024 = next(arr for d, arr in openai_result["total_h100e_by_date"].items()
                       if d.year == 2024)
    backcasts = {
        "Google DeepMind": model_deepmind_2024(),
        "Meta AI (pre-MSL)": model_msl_2024(),
        "OpenAI": openai_2024,
        "Anthropic": model_anthropic_2024(),
        "SpaceXAI": spacexai["2024"],
    }
    print("\nEnd-2024 backcasts:\n")
    print(f"  {'Lab':<18}{'5th':>10}{'median':>10}{'95th':>10}")
    for name, samples in backcasts.items():
        lo, mid, hi = pctiles(samples)
        print(f"  {name:<18}{fmt(lo):>10}{fmt(mid):>10}{fmt(hi):>10}")

    # SpaceXAI mid-2026 (spacexai["h1_2026"], June 30 net of sales) is held back
    # from the mainline exports for now; it still lives in the
    # spacexai_compute_model notebook. model_spacexai() keeps computing it.

    # One comparison chart: median bar per lab with a 90% CI error bar.
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.13)
    colors = {"Google DeepMind": "#2B8C86", "Meta SL": "#2B6CB8",
              "OpenAI": "#1a73e8", "Anthropic": "#e8710a", "SpaceXAI": "#333333"}
    names = list(labs)
    highest = 0.0
    for i, name in enumerate(names):
        y = len(names) - 1 - i
        lo, mid, hi = pctiles(labs[name])
        highest = max(highest, hi / 1e6)
        ax.barh(y, mid / 1e6, height=0.5, color=colors[name], alpha=0.85)
        ax.errorbar(mid / 1e6, y, xerr=[[(mid - lo) / 1e6], [(hi - mid) / 1e6]],
                    fmt="none", ecolor="#333333", elinewidth=1.5, capsize=5, capthick=1.5)
        ax.text(hi / 1e6 + 0.04, y, f"{fmt(mid)}  (90% CI {fmt(lo)}-{fmt(hi)})",
                va="center", fontsize=9, color="#444444")
    ax.set_xlim(0, highest * 1.55)  # headroom so the annotations don't clip
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)))
    ax.set_xlabel("End-2025 compute (H100e, millions)")
    ax.set_title("Frontier-lab compute at end-2025", loc="left", weight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.show()

    return labs


if __name__ == "__main__":
    main()
