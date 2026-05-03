"""Unit tests for rice_Ml.preprocessing.features."""

import numpy as np
import pytest

from rice_Ml.preprocessing.features import (
    rms_energy,
    zero_crossing_rate,
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    mfcc,
    extract_all,
)

SR = 44100
FRAME_LEN = 2048
N_FRAMES = 10


def _sine_frames(freq_hz: float, n_frames: int = N_FRAMES) -> np.ndarray:
    """Generate frames filled with a single sine wave at *freq_hz*."""
    t = np.arange(FRAME_LEN) / SR
    wave = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return np.tile(wave, (n_frames, 1))


def _silence_frames(n_frames: int = N_FRAMES) -> np.ndarray:
    return np.zeros((n_frames, FRAME_LEN), dtype=np.float32)


@pytest.fixture
def sine_frames():
    return _sine_frames(440.0)


@pytest.fixture
def silence():
    return _silence_frames()


# ===========================================================================
# rms_energy
# ===========================================================================

class TestRmsEnergy:
    def test_shape(self, sine_frames):
        assert rms_energy(sine_frames).shape == (N_FRAMES,)

    def test_silence_is_zero(self, silence):
        np.testing.assert_allclose(rms_energy(silence), 0.0, atol=1e-10)

    def test_sine_nonzero(self, sine_frames):
        assert np.all(rms_energy(sine_frames) > 0)

    def test_louder_signal_higher_rms(self):
        quiet = _sine_frames(440.0) * 0.1
        loud  = _sine_frames(440.0) * 0.9
        assert np.all(rms_energy(loud) > rms_energy(quiet))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            rms_energy(np.ones(100))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rms_energy(np.zeros((0, FRAME_LEN)))


# ===========================================================================
# zero_crossing_rate
# ===========================================================================

class TestZeroCrossingRate:
    def test_shape(self, sine_frames):
        assert zero_crossing_rate(sine_frames).shape == (N_FRAMES,)

    def test_silence_is_zero(self, silence):
        np.testing.assert_allclose(zero_crossing_rate(silence), 0.0, atol=1e-10)

    def test_range_0_to_1(self, sine_frames):
        zcr = zero_crossing_rate(sine_frames)
        assert np.all(zcr >= 0.0) and np.all(zcr <= 1.0)

    def test_high_freq_more_crossings(self):
        lo = zero_crossing_rate(_sine_frames(110.0))
        hi = zero_crossing_rate(_sine_frames(1760.0))
        assert np.all(hi > lo)


# ===========================================================================
# spectral_centroid
# ===========================================================================

class TestSpectralCentroid:
    def test_shape(self, sine_frames):
        assert spectral_centroid(sine_frames, SR).shape == (N_FRAMES,)

    def test_silence_centroid_zero(self, silence):
        c = spectral_centroid(silence, SR)
        np.testing.assert_allclose(c, 0.0, atol=1.0)

    def test_centroid_positive_for_sine(self):
        c = spectral_centroid(_sine_frames(440.0), SR)
        assert np.all(c > 0.0)

    def test_higher_freq_higher_centroid(self):
        c_lo = spectral_centroid(_sine_frames(220.0), SR)
        c_hi = spectral_centroid(_sine_frames(880.0), SR)
        assert np.all(c_hi > c_lo)

    def test_invalid_sr(self, sine_frames):
        with pytest.raises(ValueError):
            spectral_centroid(sine_frames, sr=0)


# ===========================================================================
# spectral_bandwidth
# ===========================================================================

class TestSpectralBandwidth:
    def test_shape(self, sine_frames):
        assert spectral_bandwidth(sine_frames, SR).shape == (N_FRAMES,)

    def test_noise_wider_bandwidth_than_sine(self):
        # White noise energy is spread across all bins → wider bandwidth than a sine
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((N_FRAMES, FRAME_LEN)).astype(np.float32)
        bw_noise = spectral_bandwidth(noise, SR).mean()
        bw_sine  = spectral_bandwidth(_sine_frames(440.0), SR).mean()
        assert bw_noise > bw_sine

    def test_non_negative(self, sine_frames):
        assert np.all(spectral_bandwidth(sine_frames, SR) >= 0.0)


# ===========================================================================
# spectral_rolloff
# ===========================================================================

class TestSpectralRolloff:
    def test_shape(self, sine_frames):
        assert spectral_rolloff(sine_frames, SR).shape == (N_FRAMES,)

    def test_rolloff_above_sine_frequency(self):
        freq = 440.0
        ro = spectral_rolloff(_sine_frames(freq), SR, roll_percent=0.85)
        # 85 % of energy is at the sine frequency; rolloff must be >= that
        assert np.all(ro >= freq * 0.9)

    def test_invalid_roll_percent(self, sine_frames):
        with pytest.raises(ValueError):
            spectral_rolloff(sine_frames, SR, roll_percent=1.5)

    def test_non_negative(self, sine_frames):
        assert np.all(spectral_rolloff(sine_frames, SR) >= 0.0)


# ===========================================================================
# mfcc
# ===========================================================================

class TestMFCC:
    def test_shape_default(self, sine_frames):
        m = mfcc(sine_frames, SR)
        assert m.shape == (N_FRAMES, 13)

    def test_shape_custom_n_mfcc(self, sine_frames):
        m = mfcc(sine_frames, SR, n_mfcc=20, n_mels=40)
        assert m.shape == (N_FRAMES, 20)

    def test_returns_float64(self, sine_frames):
        assert mfcc(sine_frames, SR).dtype == np.float64

    def test_silence_vs_sine_different(self, sine_frames, silence):
        m_sine    = mfcc(sine_frames, SR)
        m_silence = mfcc(silence, SR)
        assert not np.allclose(m_sine, m_silence)

    def test_n_mels_less_than_n_mfcc_raises(self, sine_frames):
        with pytest.raises(ValueError, match="n_mels"):
            mfcc(sine_frames, SR, n_mfcc=20, n_mels=10)

    def test_invalid_n_mfcc(self, sine_frames):
        with pytest.raises(ValueError):
            mfcc(sine_frames, SR, n_mfcc=0)

    def test_different_frequencies_give_different_mfcc(self):
        m1 = mfcc(_sine_frames(220.0), SR)
        m2 = mfcc(_sine_frames(880.0), SR)
        assert not np.allclose(m1, m2)


# ===========================================================================
# extract_all
# ===========================================================================

class TestExtractAll:
    def test_shape_default(self, sine_frames):
        X = extract_all(sine_frames, SR)
        assert X.shape == (N_FRAMES, 18)   # 5 spectral + 13 MFCC

    def test_shape_custom_n_mfcc(self, sine_frames):
        X = extract_all(sine_frames, SR, n_mfcc=20)
        assert X.shape == (N_FRAMES, 25)   # 5 + 20

    def test_dtype_float64(self, sine_frames):
        assert extract_all(sine_frames, SR).dtype == np.float64

    def test_no_nans(self, sine_frames):
        X = extract_all(sine_frames, SR)
        assert not np.any(np.isnan(X))

    def test_no_infs(self, sine_frames):
        X = extract_all(sine_frames, SR)
        assert not np.any(np.isinf(X))

    def test_silence_no_nans(self, silence):
        X = extract_all(silence, SR)
        assert not np.any(np.isnan(X))

    def test_different_signals_different_features(self, sine_frames, silence):
        assert not np.allclose(extract_all(sine_frames, SR), extract_all(silence, SR))
