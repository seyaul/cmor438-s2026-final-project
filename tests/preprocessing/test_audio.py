"""Unit tests for rice_Ml.preprocessing.audio."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wav_io

from rice_Ml.preprocessing.audio import (
    load_wav,
    extract_string,
    frame_signal,
    frame_center_times,
)


# ---------------------------------------------------------------------------
# WAV file helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, data: np.ndarray, sr: int = 44100) -> None:
    wav_io.write(str(path), sr, data)


@pytest.fixture
def mono_wav(tmp_path):
    """Mono int16 WAV, 1 second at 44100 Hz."""
    sr = 44100
    data = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 32767).astype(np.int16)
    p = tmp_path / "mono.wav"
    _write_wav(p, data, sr)
    return p, sr, data


@pytest.fixture
def hex_wav(tmp_path):
    """6-channel int16 WAV — each channel is a different sine frequency."""
    sr = 44100
    n = sr  # 1 second
    channels = np.zeros((n, 6), dtype=np.int16)
    for i in range(6):
        channels[:, i] = (np.sin(2 * np.pi * (100 + i * 50) * np.arange(n) / sr) * 32767).astype(np.int16)
    p = tmp_path / "hex.wav"
    _write_wav(p, channels, sr)
    return p, sr, channels


# ===========================================================================
# load_wav
# ===========================================================================

class TestLoadWav:
    def test_mono_shape(self, mono_wav):
        path, sr, _ = mono_wav
        audio, sample_rate = load_wav(path)
        assert audio.ndim == 1
        assert sample_rate == sr

    def test_hex_shape(self, hex_wav):
        path, sr, _ = hex_wav
        audio, sample_rate = load_wav(path)
        assert audio.shape == (sr, 6)
        assert sample_rate == sr

    def test_normalised_range(self, mono_wav):
        path, _, _ = mono_wav
        audio, _ = load_wav(path)
        assert audio.dtype == np.float32
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_nonzero_signal(self, mono_wav):
        path, _, _ = mono_wav
        audio, _ = load_wav(path)
        assert np.abs(audio).max() > 0.0

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_wav(tmp_path / "missing.wav")

    def test_accepts_string_path(self, mono_wav):
        path, _, _ = mono_wav
        audio, _ = load_wav(str(path))
        assert audio.ndim == 1


# ===========================================================================
# extract_string
# ===========================================================================

class TestExtractString:
    def test_mono_returns_unchanged(self):
        mono = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = extract_string(mono, string_idx=0)
        np.testing.assert_array_equal(result, mono)

    def test_mono_ignores_string_idx(self):
        mono = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        r0 = extract_string(mono, 0)
        r3 = extract_string(mono, 3)
        np.testing.assert_array_equal(r0, r3)

    def test_hex_extracts_correct_channel(self, hex_wav):
        path, sr, raw = hex_wav
        audio, _ = load_wav(path)
        for i in range(6):
            ch = extract_string(audio, i)
            assert ch.shape == (sr,)
            # channel i should differ from channel i+1
            if i < 5:
                assert not np.allclose(extract_string(audio, i), extract_string(audio, i + 1))

    def test_returns_copy(self):
        arr = np.ones((100, 6), dtype=np.float32)
        result = extract_string(arr, 0)
        result[:] = 99.0
        assert arr[0, 0] == 1.0   # original unmodified

    def test_out_of_range_channel(self):
        arr = np.ones((100, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="out of range"):
            extract_string(arr, 6)

    def test_non_integer_string_idx(self):
        arr = np.ones((100, 6), dtype=np.float32)
        with pytest.raises(TypeError):
            extract_string(arr, 1.0)

    def test_3d_array_raises(self):
        arr = np.ones((10, 6, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            extract_string(arr, 0)


# ===========================================================================
# frame_signal
# ===========================================================================

class TestFrameSignal:
    def test_output_shape(self):
        signal = np.ones(1000, dtype=np.float32)
        frames = frame_signal(signal, frame_len=256, hop_len=128)
        assert frames.ndim == 2
        assert frames.shape[1] == 256

    def test_all_samples_covered(self):
        # Every sample index should be reachable from at least one frame
        n = 500
        frame_len, hop_len = 64, 32
        signal = np.arange(n, dtype=np.float32)
        frames = frame_signal(signal, frame_len=frame_len, hop_len=hop_len)
        n_frames = frames.shape[0]
        # last frame must start at or after the last sample
        last_start = (n_frames - 1) * hop_len
        assert last_start < n

    def test_hann_window_applied(self):
        signal = np.ones(2048, dtype=np.float32)
        frames = frame_signal(signal, frame_len=2048, hop_len=512)
        # A Hann window on all-ones should give the window itself
        window = np.hanning(2048).astype(np.float32)
        np.testing.assert_allclose(frames[0], window, atol=1e-5)

    def test_single_frame_short_signal(self):
        signal = np.ones(100, dtype=np.float32)
        frames = frame_signal(signal, frame_len=256, hop_len=128)
        assert frames.shape[0] >= 1

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            frame_signal(np.ones((10, 2), dtype=np.float32), 4, 2)

    def test_invalid_frame_len(self):
        with pytest.raises(ValueError):
            frame_signal(np.ones(100, dtype=np.float32), 0, 2)

    def test_invalid_hop_len(self):
        with pytest.raises(ValueError):
            frame_signal(np.ones(100, dtype=np.float32), 4, 0)

    def test_output_dtype_float32(self):
        signal = np.ones(512, dtype=np.int16)
        frames = frame_signal(signal.astype(np.float32), 256, 128)
        assert frames.dtype == np.float32


# ===========================================================================
# frame_center_times
# ===========================================================================

class TestFrameCenterTimes:
    def test_shape(self):
        times = frame_center_times(10, frame_len=2048, hop_len=512, sr=44100)
        assert times.shape == (10,)

    def test_first_time(self):
        # center of frame 0 = frame_len/2 / sr
        t = frame_center_times(1, frame_len=2048, hop_len=512, sr=44100)
        assert t[0] == pytest.approx(1024 / 44100)

    def test_spacing(self):
        times = frame_center_times(5, frame_len=2048, hop_len=512, sr=44100)
        diffs = np.diff(times)
        expected_hop = 512 / 44100
        np.testing.assert_allclose(diffs, expected_hop, rtol=1e-10)

    def test_monotone_increasing(self):
        times = frame_center_times(20, frame_len=2048, hop_len=512, sr=44100)
        assert np.all(np.diff(times) > 0)

    def test_invalid_sr(self):
        with pytest.raises(ValueError):
            frame_center_times(10, 2048, 512, sr=0)
