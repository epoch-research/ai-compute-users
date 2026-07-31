"""Chip fleet data fetched at runtime from the Epoch AI data hub.

The models used to read frozen CSV snapshots exported by the ai-chip-counts
repo; now they pull the published versions of the same estimates from
epoch.ai. Downloads are cached under .cache/ (gitignored), one file per day,
so repeated runs on the same day reuse a single download.
"""

import datetime
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

CHIP_SALES_URL = "https://epoch.ai/data/ai_chip_sales.zip"
CHIP_OWNERS_URL = "https://epoch.ai/data/ai_chip_owners.zip"
DATA_CENTERS_URL = "https://epoch.ai/data/data_centers/data_centers.zip"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

# The published CSVs renamed a few columns relative to the exports the models
# were built against; map them back so downstream code stays unchanged.
_SALES_RENAMES = {
    "H100e compute power (median)": "Compute estimate in H100e (median)",
    "H100e compute power (5th percentile)": "Compute estimate in H100e (5th percentile)",
    "H100e compute power (95th percentile)": "Compute estimate in H100e (95th percentile)",
}
_OWNERS_RENAMES = {"Number of Units (median)": "Number of Units"}


def _fetch_zip(url):
    CACHE_DIR.mkdir(exist_ok=True)
    cached = CACHE_DIR / f"{datetime.date.today():%Y-%m-%d}_{Path(url).name}"
    if not cached.exists():
        with urllib.request.urlopen(url) as response:
            cached.write_bytes(response.read())
    return cached


@lru_cache(maxsize=None)
def _read_csv_from_zip(url, member):
    with zipfile.ZipFile(_fetch_zip(url)) as archive, archive.open(member) as f:
        return pd.read_csv(f)


def load_chip_sales_cumulative(manufacturer):
    """Cumulative units and H100e per chip type for one chip designer
    (e.g. "Google", "AMD"), quarterly, from Epoch's AI Chip Sales data."""
    df = _read_csv_from_zip(CHIP_SALES_URL, "cumulative_timelines.csv")
    df = df[df["Chip manufacturer"] == manufacturer].copy()
    df = df.rename(columns=_SALES_RENAMES)
    df["End date"] = pd.to_datetime(df["End date"])
    return df


def load_data_center_timelines():
    """Dated construction/capacity milestones per data center, from Epoch's
    AI Data Centers data: one row per (data center, date) with the estimated
    operational H100e, IT power, and construction status at that date."""
    df = _read_csv_from_zip(DATA_CENTERS_URL, "data_center_timelines.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_nvidia_owners_cumulative():
    """Cumulative Nvidia fleet per owner and chip type, quarterly, from
    Epoch's AI Chip Owners data. The published data also assigns TPU and AMD
    chips to owners; those are dropped here because the models bring in TPU
    and AMD fleets separately (keeping them would double-count)."""
    df = _read_csv_from_zip(CHIP_OWNERS_URL, "cumulative_by_chip_type.csv")
    df = df[df["Chip manufacturer"] == "Nvidia"].copy()
    df = df.rename(columns=_OWNERS_RENAMES)
    df["End date"] = pd.to_datetime(df["End date"])
    return df
