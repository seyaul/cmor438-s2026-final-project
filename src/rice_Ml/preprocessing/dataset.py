"""
Dataset assembly: label assignment and persistence.

Combines parsed JAMS annotations, loaded audio, extracted features, and
frame timestamps into labeled (X, y) NumPy arrays ready for any downstream
machine learning pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .guitarset import load_jams, get_note_events, get_pitch_contour
from .audio import load_wav, extract_string, frame_signal, frame_center_times
from .features import extract_all

_VALID_LABELS = ("midi", "voiced", "frequency")


# ---------------------------------------------------------------------------
# Label functions
# ---------------------------------------------------------------------------

def label_frames_midi(
    frame_times: np.ndarray,
    note_events: list[tuple[float, float, float]],
    silence_label: int = 0,
) -> np.ndarray:
    """Assign a rounded MIDI note number to each frame.

    For each frame whose centre time falls within a note event
    ``(onset ≤ t < offset)``, the label is ``round(midi_note)``.
    Frames outside all note events receive *silence_label* (default 0).

    Parameters
    ----------
    frame_times : numpy.ndarray, shape (n_frames,)
        Centre timestamps of each frame in seconds.
    note_events : list of (onset, offset, midi_note) tuples
        As returned by :func:`~rice_Ml.preprocessing.guitarset.get_note_events`.
    silence_label : int, default 0
        Label assigned to silent frames.

    Returns
    -------
    numpy.ndarray, shape (n_frames,), dtype int32
    """
    if frame_times.ndim != 1:
        raise ValueError("frame_times must be 1-D.")
    labels = np.full(len(frame_times), silence_label, dtype=np.int32)
    for onset, offset, midi in note_events:
        mask = (frame_times >= onset) & (frame_times < offset)
        labels[mask] = int(round(midi))
    return labels


def label_frames_voiced(
    frame_times: np.ndarray,
    contour_times: np.ndarray,
    contour_voiced: np.ndarray,
) -> np.ndarray:
    """Assign a voiced/unvoiced boolean label to each frame.

    Each frame is matched to the nearest pitch-contour observation and
    inherits that observation's ``voiced`` flag.

    Parameters
    ----------
    frame_times : numpy.ndarray, shape (n_frames,)
    contour_times : numpy.ndarray, shape (n_points,)
    contour_voiced : numpy.ndarray of bool, shape (n_points,)

    Returns
    -------
    numpy.ndarray, shape (n_frames,), dtype bool
    """
    _validate_contour_args(frame_times, contour_times, contour_voiced)
    indices = _nearest_contour_indices(frame_times, contour_times)
    return contour_voiced[indices].astype(bool)


def label_frames_frequency(
    frame_times: np.ndarray,
    contour_times: np.ndarray,
    contour_frequencies: np.ndarray,
    contour_voiced: np.ndarray,
    unvoiced_fill: float = 0.0,
) -> np.ndarray:
    """Assign the nearest pitch-contour frequency (Hz) to each frame.

    Unvoiced frames receive *unvoiced_fill* (default 0.0).

    Parameters
    ----------
    frame_times : numpy.ndarray, shape (n_frames,)
    contour_times : numpy.ndarray, shape (n_points,)
    contour_frequencies : numpy.ndarray, shape (n_points,)
        Frequency values in Hz.
    contour_voiced : numpy.ndarray of bool, shape (n_points,)
    unvoiced_fill : float, default 0.0

    Returns
    -------
    numpy.ndarray, shape (n_frames,), dtype float64
    """
    _validate_contour_args(frame_times, contour_times, contour_voiced)
    if contour_frequencies.shape != contour_times.shape:
        raise ValueError(
            "contour_frequencies and contour_times must have the same shape."
        )
    indices = _nearest_contour_indices(frame_times, contour_times)
    freqs = contour_frequencies[indices].astype(np.float64)
    voiced = contour_voiced[indices]
    freqs[~voiced] = float(unvoiced_fill)
    return freqs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    hex_cln_wav_path: str | Path,
    jams_path: str | Path,
    string_idx: int = 0,
    frame_len: int = 2048,
    hop_len: int = 512,
    label: str = "midi",
    n_mfcc: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full preprocessing pipeline for one guitar string.

    Steps:

    1. Parse the JAMS file → note events and pitch contour for *string_idx*.
    2. Load the ``hex_cln`` WAV → extract the channel for *string_idx*.
    3. Frame the signal and compute centre timestamps.
    4. Extract all audio features → ``X`` of shape ``(n_frames, n_mfcc + 5)``.
    5. Assign frame labels → ``y`` of shape ``(n_frames,)``.

    Parameters
    ----------
    hex_cln_wav_path : str or Path
        Path to the 6-channel ``hex_cln`` WAV file for this excerpt.
    jams_path : str or Path
        Path to the corresponding ``.jams`` annotation file.
    string_idx : int, default 0
        Guitar string to process (0 = low E, 5 = high E).
    frame_len : int, default 2048
        Samples per frame (~46 ms at 44.1 kHz).
    hop_len : int, default 512
        Samples between frame starts (~12 ms hop, 75 % overlap).
    label : {"midi", "voiced", "frequency"}, default ``"midi"``
        Label type:

        - ``"midi"``      — rounded MIDI note number (int32, 0 = silence)
        - ``"voiced"``    — voiced/unvoiced flag (bool)
        - ``"frequency"`` — fundamental frequency in Hz (float64, 0.0 = silence)
    n_mfcc : int, default 13
        Number of MFCC coefficients.

    Returns
    -------
    X : numpy.ndarray, shape (n_frames, n_mfcc + 5)
        Feature matrix (default 18 columns).
    y : numpy.ndarray, shape (n_frames,)
        Label array.

    Raises
    ------
    ValueError
        If *label* is not one of the supported values, or any path is invalid.
    """
    if label not in _VALID_LABELS:
        raise ValueError(
            f"label must be one of {_VALID_LABELS}, got {label!r}."
        )

    # --- annotations ---
    jams = load_jams(jams_path)
    note_events = get_note_events(jams, string_idx)
    c_times, c_freqs, c_voiced = get_pitch_contour(jams, string_idx)

    # --- audio ---
    audio, sr = load_wav(hex_cln_wav_path)
    mono = extract_string(audio, string_idx)
    frames = frame_signal(mono, frame_len=frame_len, hop_len=hop_len)
    n_frames = frames.shape[0]
    times = frame_center_times(n_frames, frame_len=frame_len, hop_len=hop_len, sr=sr)

    # --- features ---
    X = extract_all(frames, sr=sr, n_mfcc=n_mfcc)

    # --- labels ---
    if label == "midi":
        y = label_frames_midi(times, note_events)
    elif label == "voiced":
        y = label_frames_voiced(times, c_times, c_voiced)
    else:  # "frequency"
        y = label_frames_frequency(times, c_times, c_freqs, c_voiced)

    return X, y


def build_multi_string_dataset(
    hex_cln_wav_path: str | Path,
    jams_path: str | Path,
    string_indices: Sequence[int] | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a dataset by stacking results across multiple strings.

    Parameters
    ----------
    hex_cln_wav_path : str or Path
    jams_path : str or Path
    string_indices : sequence of int or None, default None
        Which strings to include.  ``None`` uses all 6 strings (0–5).
    **kwargs
        Forwarded to :func:`build_dataset` (e.g. ``label``, ``frame_len``).

    Returns
    -------
    X : numpy.ndarray, shape (n_total_frames, n_features)
    y : numpy.ndarray, shape (n_total_frames,)
    """
    if string_indices is None:
        string_indices = list(range(6))

    X_parts, y_parts = [], []
    for idx in string_indices:
        X_i, y_i = build_dataset(
            hex_cln_wav_path, jams_path, string_idx=idx, **kwargs
        )
        X_parts.append(X_i)
        y_parts.append(y_i)

    return np.vstack(X_parts), np.concatenate(y_parts)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_dataset(X: np.ndarray, y: np.ndarray, path: str | Path) -> None:
    """Save *(X, y)* to a compressed NumPy archive (``.npz``).

    Parameters
    ----------
    X : numpy.ndarray, shape (n_frames, n_features)
    y : numpy.ndarray, shape (n_frames,)
    path : str or Path
        Output file path.  The ``.npz`` extension is appended automatically
        if not present.
    """
    path = Path(path)
    np.savez_compressed(str(path), X=X, y=y)


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load *(X, y)* from a compressed NumPy archive saved by :func:`save_dataset`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file (with or without the extension).

    Returns
    -------
    X : numpy.ndarray
    y : numpy.ndarray

    Raises
    ------
    FileNotFoundError
        If the archive does not exist.
    """
    path = Path(path)
    # np.load appends .npz automatically if missing, but we want a clear error
    candidate = path if path.suffix == ".npz" else path.with_suffix(".npz")
    if not candidate.exists() and not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    data = np.load(str(path) if path.exists() else str(candidate))
    return data["X"], data["y"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_contour_args(
    frame_times: np.ndarray,
    contour_times: np.ndarray,
    contour_voiced: np.ndarray,
) -> None:
    if frame_times.ndim != 1:
        raise ValueError("frame_times must be 1-D.")
    if contour_times.ndim != 1:
        raise ValueError("contour_times must be 1-D.")
    if contour_voiced.shape != contour_times.shape:
        raise ValueError(
            "contour_voiced and contour_times must have the same shape."
        )
    if len(contour_times) == 0:
        raise ValueError("contour_times must not be empty.")


def _nearest_contour_indices(
    frame_times: np.ndarray, contour_times: np.ndarray
) -> np.ndarray:
    """Return index into *contour_times* nearest to each value in *frame_times*."""
    # searchsorted is O(n log m) — fast enough for ~4 k contour points
    idx = np.searchsorted(contour_times, frame_times)
    idx = np.clip(idx, 0, len(contour_times) - 1)
    # check if the previous index is closer
    prev = np.clip(idx - 1, 0, len(contour_times) - 1)
    closer_prev = np.abs(frame_times - contour_times[prev]) < np.abs(
        frame_times - contour_times[idx]
    )
    idx = np.where(closer_prev, prev, idx)
    return idx
