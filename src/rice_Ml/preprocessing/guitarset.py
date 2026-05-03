"""
GuitarSet JAMS file parser.

Loads .jams annotation files (plain JSON) and extracts per-string note events
and pitch contour data for use in the preprocessing pipeline.

JAMS annotation array layout (confirmed from GuitarSet 0.3.1):
    indices 0–5   → pitch_contour  (one per string)
    indices 6–11  → note_midi      (one per string)
    index   12    → beat_position
    index   13    → tempo
    indices 14–15 → chord (instructed, performed)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_N_STRINGS = 6
_PITCH_CONTOUR_OFFSET = 0   # pitch_contour annotations start at index 0
_NOTE_MIDI_OFFSET = 6       # note_midi annotations start at index 6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jams(path: str | Path) -> dict:
    """Load a GuitarSet JAMS file and return the raw parsed dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the ``.jams`` file.

    Returns
    -------
    dict
        Top-level JAMS dictionary with keys ``annotations``, ``file_metadata``,
        and ``sandbox``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JAMS file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {path} as JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def get_duration(jams: dict) -> float:
    """Return the recording duration in seconds.

    Parameters
    ----------
    jams : dict
        Parsed JAMS dictionary from :func:`load_jams`.

    Returns
    -------
    float
    """
    return float(jams["file_metadata"]["duration"])


def get_tempo(jams: dict) -> float:
    """Return the tempo in BPM.

    Parameters
    ----------
    jams : dict
        Parsed JAMS dictionary from :func:`load_jams`.

    Returns
    -------
    float
    """
    tempo_annotation = _find_annotation(jams, "tempo")
    return float(tempo_annotation["data"][0]["value"])


# ---------------------------------------------------------------------------
# Per-string annotation extractors
# ---------------------------------------------------------------------------

def get_note_events(
    jams: dict, string_idx: int
) -> list[tuple[float, float, float]]:
    """Extract discrete note events for one guitar string.

    Parameters
    ----------
    jams : dict
        Parsed JAMS dictionary from :func:`load_jams`.
    string_idx : int
        Guitar string index in ``[0, 5]`` (0 = low E, 5 = high E).

    Returns
    -------
    list of (onset_sec, offset_sec, midi_note) tuples
        ``onset_sec`` and ``offset_sec`` are floats in seconds.
        ``midi_note`` is a continuous float (e.g. 39.97 ≈ MIDI 40).
        Sorted by onset time.

    Raises
    ------
    ValueError
        If *string_idx* is outside ``[0, 5]``.
    """
    _validate_string_idx(string_idx)
    annotation = jams["annotations"][_NOTE_MIDI_OFFSET + string_idx]
    events = []
    for entry in annotation["data"]:
        onset = float(entry["time"])
        duration = float(entry["duration"])
        midi = float(entry["value"])
        events.append((onset, onset + duration, midi))
    events.sort(key=lambda e: e[0])
    return events


def get_pitch_contour(
    jams: dict, string_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the continuous pitch contour for one guitar string.

    Parameters
    ----------
    jams : dict
        Parsed JAMS dictionary from :func:`load_jams`.
    string_idx : int
        Guitar string index in ``[0, 5]``.

    Returns
    -------
    times : numpy.ndarray, shape (n_points,)
        Timestamps in seconds for each contour observation.
    frequencies : numpy.ndarray, shape (n_points,)
        Fundamental frequency in Hz at each timestamp.
        Zero for unvoiced frames.
    voiced : numpy.ndarray of bool, shape (n_points,)
        ``True`` where a pitch is detected (voiced); ``False`` for silence.

    Raises
    ------
    ValueError
        If *string_idx* is outside ``[0, 5]``.
    """
    _validate_string_idx(string_idx)
    annotation = jams["annotations"][_PITCH_CONTOUR_OFFSET + string_idx]
    data = annotation["data"]

    times = np.array([float(e["time"]) for e in data], dtype=np.float64)
    voiced = np.array([bool(e["value"]["voiced"]) for e in data], dtype=bool)
    frequencies = np.array(
        [float(e["value"]["frequency"]) if e["value"]["voiced"] else 0.0
         for e in data],
        dtype=np.float64,
    )
    return times, frequencies, voiced


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_string_idx(string_idx: int) -> None:
    if not isinstance(string_idx, int) or isinstance(string_idx, bool):
        raise TypeError(
            f"string_idx must be an integer, got {type(string_idx).__name__!r}."
        )
    if not (0 <= string_idx < _N_STRINGS):
        raise ValueError(
            f"string_idx must be in [0, {_N_STRINGS - 1}], got {string_idx}."
        )


def _find_annotation(jams: dict, namespace: str) -> dict:
    """Return the first annotation whose namespace matches *namespace*."""
    for ann in jams["annotations"]:
        if ann.get("namespace") == namespace:
            return ann
    raise KeyError(f"No annotation with namespace {namespace!r} found.")
