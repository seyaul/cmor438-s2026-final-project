"""Unit tests for rice_Ml.preprocessing.dataset."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rice_Ml.preprocessing.dataset import (
    label_frames_midi,
    label_frames_voiced,
    label_frames_frequency,
    save_dataset,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def frame_times():
    return np.linspace(0.0, 2.0, 20)


@pytest.fixture
def note_events():
    # note 40 from t=0.5 to t=1.0, note 45 from t=1.2 to t=1.8
    return [(0.5, 1.0, 40.0), (1.2, 1.8, 45.3)]


@pytest.fixture
def contour(frame_times):
    times = np.linspace(0.0, 2.0, 50)
    voiced = times > 0.5
    freqs = np.where(voiced, 82.4, 0.0)
    return times, freqs, voiced


# ===========================================================================
# label_frames_midi
# ===========================================================================

class TestLabelFramesMidi:
    def test_shape(self, frame_times, note_events):
        y = label_frames_midi(frame_times, note_events)
        assert y.shape == (len(frame_times),)

    def test_dtype_int32(self, frame_times, note_events):
        y = label_frames_midi(frame_times, note_events)
        assert y.dtype == np.int32

    def test_silence_label_default_zero(self, frame_times, note_events):
        y = label_frames_midi(frame_times, note_events)
        # frames before t=0.5 should be silence (0)
        early_mask = frame_times < 0.5
        assert np.all(y[early_mask] == 0)

    def test_active_note_label(self, frame_times, note_events):
        y = label_frames_midi(frame_times, note_events)
        # frames in [0.5, 1.0) should be labeled 40
        active = (frame_times >= 0.5) & (frame_times < 1.0)
        assert np.all(y[active] == 40)

    def test_second_note_label(self, frame_times, note_events):
        y = label_frames_midi(frame_times, note_events)
        active = (frame_times >= 1.2) & (frame_times < 1.8)
        # round(45.3) = 45
        assert np.all(y[active] == 45)

    def test_custom_silence_label(self, frame_times, note_events):
        y = label_frames_midi(frame_times, note_events, silence_label=-1)
        early = frame_times < 0.5
        assert np.all(y[early] == -1)

    def test_empty_note_events(self, frame_times):
        y = label_frames_midi(frame_times, [])
        assert np.all(y == 0)

    def test_non_1d_frame_times_raises(self, note_events):
        with pytest.raises(ValueError):
            label_frames_midi(np.zeros((5, 2)), note_events)


# ===========================================================================
# label_frames_voiced
# ===========================================================================

class TestLabelFramesVoiced:
    def test_shape(self, frame_times, contour):
        c_times, _, c_voiced = contour
        y = label_frames_voiced(frame_times, c_times, c_voiced)
        assert y.shape == (len(frame_times),)

    def test_dtype_bool(self, frame_times, contour):
        c_times, _, c_voiced = contour
        y = label_frames_voiced(frame_times, c_times, c_voiced)
        assert y.dtype == bool

    def test_early_frames_unvoiced(self, frame_times, contour):
        c_times, _, c_voiced = contour
        y = label_frames_voiced(frame_times, c_times, c_voiced)
        early = frame_times <= 0.5
        assert not np.any(y[early])

    def test_late_frames_voiced(self, frame_times, contour):
        c_times, _, c_voiced = contour
        y = label_frames_voiced(frame_times, c_times, c_voiced)
        late = frame_times > 0.6
        assert np.all(y[late])

    def test_mismatched_contour_shapes_raises(self, frame_times):
        times = np.array([0.0, 1.0, 2.0])
        voiced = np.array([True, False])   # wrong length
        with pytest.raises(ValueError):
            label_frames_voiced(frame_times, times, voiced)

    def test_empty_contour_raises(self, frame_times):
        with pytest.raises(ValueError, match="empty"):
            label_frames_voiced(frame_times, np.array([]), np.array([], dtype=bool))


# ===========================================================================
# label_frames_frequency
# ===========================================================================

class TestLabelFramesFrequency:
    def test_shape(self, frame_times, contour):
        c_times, c_freqs, c_voiced = contour
        y = label_frames_frequency(frame_times, c_times, c_freqs, c_voiced)
        assert y.shape == (len(frame_times),)

    def test_dtype_float64(self, frame_times, contour):
        c_times, c_freqs, c_voiced = contour
        y = label_frames_frequency(frame_times, c_times, c_freqs, c_voiced)
        assert y.dtype == np.float64

    def test_unvoiced_frames_zero(self, frame_times, contour):
        c_times, c_freqs, c_voiced = contour
        y = label_frames_frequency(frame_times, c_times, c_freqs, c_voiced)
        early = frame_times <= 0.5
        np.testing.assert_allclose(y[early], 0.0, atol=1e-6)

    def test_voiced_frames_nonzero(self, frame_times, contour):
        c_times, c_freqs, c_voiced = contour
        y = label_frames_frequency(frame_times, c_times, c_freqs, c_voiced)
        late = frame_times > 0.6
        assert np.all(y[late] > 0.0)

    def test_custom_unvoiced_fill(self, frame_times, contour):
        c_times, c_freqs, c_voiced = contour
        y = label_frames_frequency(frame_times, c_times, c_freqs, c_voiced,
                                    unvoiced_fill=-1.0)
        early = frame_times <= 0.5
        np.testing.assert_allclose(y[early], -1.0, atol=1e-6)

    def test_mismatched_freq_shape_raises(self, frame_times, contour):
        c_times, _, c_voiced = contour
        bad_freqs = np.ones(len(c_times) + 5)
        with pytest.raises(ValueError):
            label_frames_frequency(frame_times, c_times, bad_freqs, c_voiced)


# ===========================================================================
# save_dataset / load_dataset
# ===========================================================================

class TestPersistence:
    def _make_xy(self):
        X = np.random.default_rng(0).standard_normal((100, 18)).astype(np.float64)
        y = np.arange(100, dtype=np.int32)
        return X, y

    def test_round_trip(self, tmp_path):
        X, y = self._make_xy()
        path = tmp_path / "dataset"
        save_dataset(X, y, path)
        X2, y2 = load_dataset(str(path) + ".npz")
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(y, y2)

    def test_load_with_npz_extension(self, tmp_path):
        X, y = self._make_xy()
        path = tmp_path / "ds.npz"
        save_dataset(X, y, path)
        X2, y2 = load_dataset(path)
        np.testing.assert_array_equal(X, X2)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.npz")

    def test_shapes_preserved(self, tmp_path):
        X, y = self._make_xy()
        path = tmp_path / "shapes"
        save_dataset(X, y, path)
        X2, y2 = load_dataset(str(path) + ".npz")
        assert X2.shape == (100, 18)
        assert y2.shape == (100,)
