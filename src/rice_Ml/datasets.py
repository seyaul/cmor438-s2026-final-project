"""
GuitarSet feature dataset loader.

Downloads `guitarset_subset.csv.gz` from the GitHub Release on first use
and caches it at ~/.cache/rice_ml/. Subsequent calls load from cache.

Usage
-----
    from rice_Ml.datasets import load_guitarset
    df = load_guitarset()          # returns the full subset DataFrame
"""

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

_SUBSET_URL = (
    "https://github.com/seyaul/cmor438-s2026-final-project"
    "/releases/download/v1.0-data/guitarset_subset.csv.gz"
)
_SUBSET_SHA256 = "9f92ae1671f07b1a44bc502405346a50bc56a205c3a00da00f24d0931f02a0f3"

_CACHE_DIR = Path.home() / ".cache" / "rice_ml"
_SUBSET_CACHE = _CACHE_DIR / "guitarset_subset.csv.gz"

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


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} from GitHub Release … ", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    print("done.")


def load_guitarset(force_download: bool = False) -> pd.DataFrame:
    """Return the GuitarSet subset as a pandas DataFrame.

    Columns: track_id, string_idx, rms, zcr, centroid, bandwidth, rolloff,
             mfcc_1 … mfcc_13, midi_label

    Parameters
    ----------
    force_download : bool
        If True, re-download even if a cached copy exists.

    Returns
    -------
    pd.DataFrame
        ~150 K rows × 20 columns. Filter/transform as needed per algorithm:
        - Perceptron: filter string_idx==0, binary voiced = (midi_label != 0)
        - MLP: drop midi_label==0, keep top-N notes across all strings
    """
    if force_download and _SUBSET_CACHE.exists():
        _SUBSET_CACHE.unlink()

    if not _SUBSET_CACHE.exists():
        _download(_SUBSET_URL, _SUBSET_CACHE)
        if not _verify_sha256(_SUBSET_CACHE, _SUBSET_SHA256):
            _SUBSET_CACHE.unlink()
            raise RuntimeError(
                "SHA-256 mismatch on downloaded file — try again or check the release."
            )
    else:
        print(f"Loading from cache: {_SUBSET_CACHE}")

    return pd.read_csv(_SUBSET_CACHE)
