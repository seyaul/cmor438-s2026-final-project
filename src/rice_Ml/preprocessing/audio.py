"""
Audio loading and framing utilities.

Handles multi-channel hexaphonic WAV files (``hex_cln``) where each of the
6 channels corresponds to one guitar string, as well as standard mono WAV
files (``mic`` recordings).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav

_N_STRINGS = 6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file and return a normalised float32 array.

    For GuitarSet ``hex_cln`` files the returned array has shape
    ``(n_samples, 6)`` — one column per string.  For mono or stereo files
    the shape is ``(n_samples,)`` or ``(n_samples, n_channels)`` respectively.

    Integer sample types (int16, int32) are normalised to ``[-1.0, 1.0]``.
    Files already stored as float32/float64 are returned unchanged (clipped to
    ``[-1.0, 1.0]``).

    Parameters
    ----------
    path : str or Path
        Path to the ``.wav`` file.

    Returns
    -------
    audio : numpy.ndarray, dtype float32
        Normalised audio samples.
    sample_rate : int
        Sample rate in Hz.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be read as a WAV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")
    try:
        sr, data = wav.read(str(path))
    except Exception as exc:
        raise ValueError(f"Could not read WAV file {path}: {exc}") from exc

    data = _normalise(data)
    return data, int(sr)


def _normalise(data: np.ndarray) -> np.ndarray:
    """Convert integer PCM samples to float32 in [-1.0, 1.0]."""
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        max_abs = float(max(abs(info.min), info.max))
        float_data = data.astype(np.float32) / max_abs
        return np.clip(float_data, -1.0, 1.0)
    float_data = data.astype(np.float32)
    return np.clip(float_data, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Channel extraction
# ---------------------------------------------------------------------------

def extract_string(audio: np.ndarray, string_idx: int) -> np.ndarray:
    """Extract the mono audio for a single guitar string.

    Parameters
    ----------
    audio : numpy.ndarray
        Audio array returned by :func:`load_wav`.

        - Shape ``(n_samples, 6)`` for ``hex_cln`` files — *string_idx*
          selects the channel.
        - Shape ``(n_samples,)`` for mono files — *string_idx* is ignored
          and the full array is returned.

    string_idx : int
        Guitar string index in ``[0, 5]`` (0 = low E, 5 = high E).
        Ignored for mono audio.

    Returns
    -------
    numpy.ndarray, shape (n_samples,), dtype float32
        Mono audio for the requested string.

    Raises
    ------
    TypeError
        If *string_idx* is not an integer.
    ValueError
        If *string_idx* is out of range for multi-channel audio, or if
        *audio* has an unexpected number of dimensions.
    """
    if not isinstance(string_idx, int) or isinstance(string_idx, bool):
        raise TypeError(
            f"string_idx must be an integer, got {type(string_idx).__name__!r}."
        )
    if audio.ndim == 1:
        # Mono — string_idx is irrelevant
        return audio.copy()
    if audio.ndim == 2:
        n_channels = audio.shape[1]
        if not (0 <= string_idx < n_channels):
            raise ValueError(
                f"string_idx {string_idx} is out of range for audio with "
                f"{n_channels} channel(s)."
            )
        return audio[:, string_idx].copy()
    raise ValueError(
        f"audio must be 1-D or 2-D, got shape {audio.shape}."
    )


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

def frame_signal(
    signal: np.ndarray,
    frame_len: int,
    hop_len: int,
) -> np.ndarray:
    """Slice a 1-D signal into overlapping Hann-weighted frames.

    The signal is zero-padded at the end so that the final sample is covered
    by at least one frame.

    Parameters
    ----------
    signal : numpy.ndarray, shape (n_samples,)
        1-D mono audio signal.
    frame_len : int
        Number of samples per frame (e.g. 2048).
    hop_len : int
        Number of samples between successive frame starts (e.g. 512).

    Returns
    -------
    frames : numpy.ndarray, shape (n_frames, frame_len), dtype float32
        Each row is one Hann-weighted audio frame.

    Raises
    ------
    ValueError
        If *signal* is not 1-D, *frame_len* < 1, or *hop_len* < 1.
    """
    if signal.ndim != 1:
        raise ValueError(
            f"signal must be 1-D, got shape {signal.shape}."
        )
    if frame_len < 1:
        raise ValueError(f"frame_len must be >= 1, got {frame_len}.")
    if hop_len < 1:
        raise ValueError(f"hop_len must be >= 1, got {hop_len}.")

    # Pad so every sample is included in at least one frame
    n_samples = len(signal)
    n_frames = 1 + max(0, (n_samples - frame_len + hop_len - 1) // hop_len)
    pad_length = (n_frames - 1) * hop_len + frame_len - n_samples
    padded = np.pad(signal.astype(np.float32), (0, pad_length))

    # Build frame matrix via strided view
    shape = (n_frames, frame_len)
    strides = (padded.strides[0] * hop_len, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides).copy()

    # Apply Hann window
    window = np.hanning(frame_len).astype(np.float32)
    frames *= window
    return frames


def frame_center_times(
    n_frames: int,
    frame_len: int,
    hop_len: int,
    sr: int,
) -> np.ndarray:
    """Return the centre timestamp (seconds) for each frame.

    Parameters
    ----------
    n_frames : int
        Total number of frames (as returned by :func:`frame_signal`).
    frame_len : int
        Samples per frame.
    hop_len : int
        Samples between frame starts.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    numpy.ndarray, shape (n_frames,), dtype float64
        ``times[i] = (i * hop_len + frame_len / 2) / sr``

    Raises
    ------
    ValueError
        If any argument is non-positive.
    """
    for name, val in [("n_frames", n_frames), ("frame_len", frame_len),
                      ("hop_len", hop_len), ("sr", sr)]:
        if val < 1:
            raise ValueError(f"{name} must be >= 1, got {val}.")
    indices = np.arange(n_frames, dtype=np.float64)
    return (indices * hop_len + frame_len / 2.0) / sr
