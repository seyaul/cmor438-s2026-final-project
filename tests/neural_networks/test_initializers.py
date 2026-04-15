import numpy as np
import pytest
from rice_ML.neural_networks.initializers import (
    Zeros, Ones, RandomNormal, RandomUniform,
    XavierUniform, XavierNormal, HeUniform, HeNormal
)

class TestInitializers:
    def test_zeros(self):
        init = Zeros()
        arr = init((3, 4))
        assert arr.shape == (3, 4)
        assert np.all(arr == 0.0)

    def test_ones(self):
        init = Ones()
        arr = init((2, 5))
        assert np.all(arr == 1.0)

    def test_random_normal_stats(self):
        init = RandomNormal(mean=1.0, stddev=0.1)
        arr = init((1000, 100))
        assert np.isclose(np.mean(arr), 1.0, atol=0.01)
        assert np.isclose(np.std(arr), 0.1, atol=0.01)

    def test_random_uniform_range(self):
        init = RandomUniform(minval=-1.0, maxval=2.0)
        arr = init((1000,))
        assert np.min(arr) >= -1.0
        assert np.max(arr) <= 2.0

    def test_xavier_uniform_scale(self):
        init = XavierUniform()
        shape = (100, 200)
        arr = init(shape)
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        assert np.min(arr) >= -limit
        assert np.max(arr) <= limit

    def test_he_uniform_scale(self):
        init = HeUniform()
        shape = (50, 100)
        arr = init(shape)
        limit = np.sqrt(6.0 / shape[0])
        assert np.min(arr) >= -limit
        assert np.max(arr) <= limit

    def test_xavier_normal_variance(self):
        init = XavierNormal()
        shape = (200, 300)
        arr = init(shape)
        expected_std = np.sqrt(2.0 / (shape[0] + shape[1]))
        assert np.isclose(np.std(arr), expected_std, rtol=0.1)

    def test_he_normal_variance(self):
        init = HeNormal()
        shape = (200, 300)
        arr = init(shape)
        expected_std = np.sqrt(2.0 / shape[0])
        assert np.isclose(np.std(arr), expected_std, rtol=0.1)