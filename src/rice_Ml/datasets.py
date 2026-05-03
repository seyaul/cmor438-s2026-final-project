"""
GuitarSet feature dataset loader.

Downloads from the GitHub Release on first use and caches at ~/.cache/rice_ml/.
Subsequent calls load from cache — no internet required.

Usage
-----
    from rice_Ml.datasets import load_guitarset

    df = load_guitarset()              # 10-track subset (~19 MB download)
    df = load_guitarset(subset=False)  # full 360-track dataset (~855 MB download)
"""

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

_BASE_URL = (
    "https://github.com/seyaul/cmor438-s2026-final-project"
    "/releases/download/v1.0-data/"
)

_FILES = {
    "subset": {
        "name": "guitarset_subset.csv.gz",
        "sha256": "9f92ae1671f07b1a44bc502405346a50bc56a205c3a00da00f24d0931f02a0f3",
        "size_hint": "~19 MB",
        "description": "10-track subset (2 per genre, randomised players, seed=42)",
    },
    "full": {
        "name": "guitarset_full.csv.gz",
        "sha256": "71d5f7ab90c56e01369170ccd8ee4f58366268540e4c3fd66b1589864b49e9f0",
        "size_hint": "~855 MB",
        "description": "Full dataset — all 360 tracks × 6 players",
    },
}

_CACHE_DIR = Path.home() / ".cache" / "rice_ml"

FEATURE_COLS = [
    "rms", "zcr", "centroid", "bandwidth", "rolloff",
    *[f"mfcc_{i}" for i in range(1, 14)],
]


def _verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def _download(url: str, dest: Path, size_hint: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ({size_hint}) from GitHub Release … ", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    print("done.")


def load_guitarset(subset: bool = True, force_download: bool = False) -> pd.DataFrame:
    """Return GuitarSet features as a pandas DataFrame.

    Parameters
    ----------
    subset : bool, default True
        True  → 10-track subset (~19 MB download, ~125 K rows).
        False → full 360-track dataset (~855 MB download, ~5.6 M rows).
    force_download : bool, default False
        Re-download even if a cached copy exists.

    Returns
    -------
    pd.DataFrame
        Columns: track_id, string_idx, rms, zcr, centroid, bandwidth, rolloff,
                 mfcc_1 … mfcc_13, midi_label

    Notes
    -----
    Common wrangling patterns::

        # Perceptron — string 0 only, binary voiced/silent
        df_s0 = df[df["string_idx"] == 0]
        y = (df_s0["midi_label"] != 0).astype(int)

        # MLP — all strings, drop silence, keep top-12 notes
        df_voiced = df[df["midi_label"] != 0]
        top12 = df_voiced["midi_label"].value_counts().head(12).index
        df_12 = df_voiced[df_voiced["midi_label"].isin(top12)]
    """
    key = "subset" if subset else "full"
    meta = _FILES[key]
    cache_path = _CACHE_DIR / meta["name"]

    if force_download and cache_path.exists():
        cache_path.unlink()

    if not cache_path.exists():
        url = _BASE_URL + meta["name"]
        _download(url, cache_path, meta["size_hint"])
        if not _verify_sha256(cache_path, meta["sha256"]):
            cache_path.unlink()
            raise RuntimeError(
                f"SHA-256 mismatch on {meta['name']} — try load_guitarset(force_download=True)."
            )
    else:
        print(f"Loading from cache: {cache_path}")

    return pd.read_csv(cache_path)
