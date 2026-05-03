"""GuitarSet access via mirdata.

Provides a thin wrapper around the mirdata GuitarSet loader so that audio and
annotations can be loaded directly without managing local file paths manually.
mirdata downloads and caches files to ~/.mirdata/guitarset/ by default.
"""
from __future__ import annotations

try:
    import mirdata
    _MIRDATA_AVAILABLE = True
except ImportError:
    _MIRDATA_AVAILABLE = False


def _require_mirdata() -> None:
    if not _MIRDATA_AVAILABLE:
        raise ImportError(
            "mirdata is required for remote GuitarSet access. "
            "Install it with: pip install 'rice_ml[data]'"
        )


def get_guitarset(data_home: str | None = None, download: bool = False):
    """Initialise the mirdata GuitarSet loader.

    GuitarSet contains five audio variants plus annotations (~7.5 GB total).
    This function only downloads the two subsets our pipeline actually uses:
    ``audio_hex_cln`` (~1.5 GB) and ``annotations`` (~37 MB).

    Parameters
    ----------
    data_home : str or None
        Local directory for cached files. Defaults to mirdata's default
        (~/.mirdata/guitarset/).
    download : bool, default False
        If True, download ``audio_hex_cln`` + ``annotations`` (~1.6 GB total).
        Set to False if you already have the files on disk.

    Returns
    -------
    guitarset : mirdata.datasets.guitarset.Dataset
    """
    _require_mirdata()
    gs = mirdata.initialize("guitarset", data_home=data_home)
    if download:
        # Only fetch the two subsets the preprocessing pipeline uses.
        # Full dataset is ~7.5 GB; audio_hex_orig / audio_mic / audio_mix
        # are not needed here.
        gs.download(partial_download=["audio_hex_cln", "annotations"])
    return gs


def track_ids(guitarset) -> list[str]:
    """Return a sorted list of all 360 track IDs in GuitarSet.

    Parameters
    ----------
    guitarset : mirdata.datasets.guitarset.Dataset
        As returned by :func:`get_guitarset`.

    Returns
    -------
    list of str
    """
    return sorted(guitarset.track_ids())
