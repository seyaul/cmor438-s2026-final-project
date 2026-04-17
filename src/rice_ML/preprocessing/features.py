"""
Frame-level audio feature extraction.

All public functions accept a 2-D frame matrix of shape ``(n_frames, frame_len)``
as produced by :func:`rice_Ml.preprocessing.audio.frame_signal` and return
NumPy arrays.  Only ``scipy`` and ``numpy`` are used — no external audio
libraries required.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct


# ---------------------------------------------------------------------------
# Time-domain features
# ---------------------------------------------------------------------------

def rms_energy(frames: np.ndarray) -> np.ndarray:
    """Root-mean-square energy per frame.

    Parameters
    ----------
    frames : numpy.ndarray, shape (n_frames, frame_len)

    Returns
    -------
    numpy.ndarray, shape (n_frames,), dtype float64
    """
    _validate_frames(frames)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Zero-crossing rate per frame.

    Counts the number of sign changes and divides by frame length so the
    result is in ``[0, 1]``.

    Parameters
    ----------
    frames : numpy.ndarray, shape (n_frames, frame_len)

    Returns
    -------
    numpy.ndarray, shape (n_frames,), dtype float64
    """
    _validate_frames(frames)
    f = frames.astype(np.float64)
    # sign changes between consecutive samples
    crossings = np.diff(np.sign(f), axis=1) != 0
    return crossings.sum(axis=1) / frames.shape[1]


# ---------------------------------------------------------------------------
# Spectral features (via FFT)
# ---------------------------------------------------------------------------

def _magnitude_spectrum(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (magnitudes, frequencies_normalised) for the positive FFT half.

    ``frequencies_normalised[k]`` is in ``[0, 0.5]`` (cycles per sample).
    """
    n = frames.shape[1]
    mag = np.abs(np.fft.rfft(frames.astype(np.float64), axis=1))
    freq = np.fft.rfftfreq(n)          # shape (n//2 + 1,)
    return mag, freq


def spectral_centroid(frames: np.ndarray, sr: int) -> np.ndarray:
    """Spectral centroid (weighted mean frequency) per frame, in Hz.

    Parameters
    ----------
    frames : numpy.ndarray, shape (n_frames, frame_len)
    sr : int
        Sample rate in Hz.

    Returns
    -------
    numpy.ndarray, shape (n_frames,), dtype float64
        Zero for frames with no energy.
    """
    _validate_frames(frames)
    _validate_sr(sr)
    mag, freq_norm = _magnitude_spectrum(frames)
    freqs_hz = freq_norm * sr                       # (n_bins,)
    energy = mag.sum(axis=1, keepdims=True)
    energy = np.where(energy == 0, 1.0, energy)     # avoid /0
    return (mag * freqs_hz).sum(axis=1) / energy.squeeze()


def spectral_bandwidth(frames: np.ndarray, sr: int) -> np.ndarray:
    """Spectral bandwidth (weighted std dev around centroid) per frame, in Hz.

    Parameters
    ----------
    frames : numpy.ndarray, shape (n_frames, frame_len)
    sr : int
        Sample rate in Hz.

    Returns
    -------
    numpy.ndarray, shape (n_frames,), dtype float64
    """
    _validate_frames(frames)
    _validate_sr(sr)
    mag, freq_norm = _magnitude_spectrum(frames)
    freqs_hz = freq_norm * sr                           # (n_bins,)
    energy = mag.sum(axis=1, keepdims=True)
    energy = np.where(energy == 0, 1.0, energy)
    centroid = (mag * freqs_hz).sum(axis=1, keepdims=True) / energy
    deviation = (freqs_hz - centroid) ** 2
    return np.sqrt((mag * deviation).sum(axis=1) / energy.squeeze())


def spectral_rolloff(
    frames: np.ndarray, sr: int, roll_percent: float = 0.85
) -> np.ndarray:
    """Spectral roll-off frequency per frame, in Hz.

    The roll-off is the frequency below which *roll_percent* of the total
    spectral energy is concentrated.

    Parameters
    ----------
    frames : numpy.ndarray, shape (n_frames, frame_len)
    sr : int
        Sample rate in Hz.
    roll_percent : float in (0, 1), default 0.85

    Returns
    -------
    numpy.ndarray, shape (n_frames,), dtype float64
    """
    _validate_frames(frames)
    _validate_sr(sr)
    if not (0.0 < roll_percent < 1.0):
        raise ValueError(
            f"roll_percent must be in (0, 1), got {roll_percent}."
        )
    mag, freq_norm = _magnitude_spectrum(frames)
    freqs_hz = freq_norm * sr
    cumsum = np.cumsum(mag, axis=1)
    total = cumsum[:, -1:]
    threshold = roll_percent * total
    # index of the first bin where cumulative energy >= threshold
    idx = np.argmax(cumsum >= threshold, axis=1)
    return freqs_hz[idx]


# ---------------------------------------------------------------------------
# MFCC
# ---------------------------------------------------------------------------

def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Build a triangular mel filterbank matrix.

    Returns
    -------
    numpy.ndarray, shape (n_mels, n_fft // 2 + 1)
        Each row is one mel filter.
    """
    n_bins = n_fft // 2 + 1
    fmin, fmax = 0.0, sr / 2.0

    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        lo, center, hi = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(lo, center):
            if center != lo:
                filterbank[m - 1, k] = (k - lo) / (center - lo)
        for k in range(center, hi):
            if hi != center:
                filterbank[m - 1, k] = (hi - k) / (hi - center)
    return filterbank


def mfcc(
    frames: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
) -> np.ndarray:
    """Mel-frequency cepstral coefficients (MFCCs) per frame.

    Parameters
    ----------
    frames : numpy.ndarray, shape (n_frames, frame_len)
    sr : int
        Sample rate in Hz.
    n_mfcc : int, default 13
        Number of cepstral coefficients to return.
    n_mels : int, default 40
        Number of mel filterbank channels.

    Returns
    -------
    numpy.ndarray, shape (n_frames, n_mfcc), dtype float64
    """
    _validate_frames(frames)
    _validate_sr(sr)
    if n_mfcc < 1:
        raise ValueError(f"n_mfcc must be >= 1, got {n_mfcc}.")
    if n_mels < n_mfcc:
        raise ValueError(
            f"n_mels ({n_mels}) must be >= n_mfcc ({n_mfcc})."
        )

    n_fft = frames.shape[1]
    mag = np.abs(np.fft.rfft(frames.astype(np.float64), axis=1))  # (n_frames, n_bins)
    fb = _mel_filterbank(n_mels, n_fft, sr)                        # (n_mels, n_bins)
    mel_energy = np.dot(mag, fb.T)                                  # (n_frames, n_mels)
    log_mel = np.log(mel_energy + 1e-10)
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")            # (n_frames, n_mels)
    return coeffs[:, :n_mfcc]


# ---------------------------------------------------------------------------
# Combined feature extractor
# ---------------------------------------------------------------------------

def extract_all(
    frames: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
) -> np.ndarray:
    """Extract all features and concatenate into a single feature matrix.

    Feature order:

    ``[rms, zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff,
    mfcc_0, mfcc_1, ..., mfcc_{n_mfcc-1}]``

    Parameters
    ----------
    frames : numpy.ndarray, shape (n_frames, frame_len)
    sr : int
        Sample rate in Hz.
    n_mfcc : int, default 13
        Number of MFCC coefficients.

    Returns
    -------
    numpy.ndarray, shape (n_frames, n_mfcc + 5), dtype float64
        Default: ``(n_frames, 18)``
    """
    _validate_frames(frames)
    _validate_sr(sr)
    cols = [
        rms_energy(frames)[:, np.newaxis],
        zero_crossing_rate(frames)[:, np.newaxis],
        spectral_centroid(frames, sr)[:, np.newaxis],
        spectral_bandwidth(frames, sr)[:, np.newaxis],
        spectral_rolloff(frames, sr)[:, np.newaxis],
        mfcc(frames, sr, n_mfcc=n_mfcc),
    ]
    return np.hstack(cols)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_frames(frames: np.ndarray) -> None:
    if not isinstance(frames, np.ndarray):
        raise TypeError(
            f"frames must be a numpy.ndarray, got {type(frames).__name__!r}."
        )
    if frames.ndim != 2:
        raise ValueError(
            f"frames must be 2-D (n_frames, frame_len), got shape {frames.shape}."
        )
    if frames.shape[0] == 0 or frames.shape[1] == 0:
        raise ValueError("frames must not be empty.")


def _validate_sr(sr: int) -> None:
    if not isinstance(sr, int) or isinstance(sr, bool) or sr < 1:
        raise ValueError(f"sr must be a positive integer, got {sr!r}.")
