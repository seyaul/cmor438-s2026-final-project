"""Unit tests for rice_Ml.measures_ml.distances."""

import math

import numpy as np
import pytest

from rice_Ml.measures_ml.distances import euclidean, taxicab


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def origin_2d():
    return [0.0, 0.0]


@pytest.fixture
def point_3_4():
    return [3.0, 4.0]


# ===========================================================================
# euclidean
# ===========================================================================

class TestEuclidean:
    """Tests for euclidean()."""

    # --- correct values ---

    def test_classic_3_4_triangle(self, origin_2d, point_3_4):
        assert euclidean(origin_2d, point_3_4) == pytest.approx(5.0)

    def test_identical_vectors_zero(self):
        assert euclidean([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_unit_vectors(self):
        # ||e1 - e2|| in R^2 = sqrt(2)
        assert euclidean([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.sqrt(2))

    def test_negative_coordinates(self):
        assert euclidean([-1, -1], [2, 3]) == pytest.approx(5.0)

    def test_high_dimensional(self):
        rng = np.random.default_rng(42)
        u = rng.standard_normal(1000)
        v = rng.standard_normal(1000)
        expected = float(np.linalg.norm(u - v))
        assert euclidean(u, v) == pytest.approx(expected, rel=1e-10)

    def test_numpy_array_input(self):
        u = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 1.0, 1.0])
        assert euclidean(u, v) == pytest.approx(math.sqrt(3))

    def test_returns_float(self, origin_2d, point_3_4):
        result = euclidean(origin_2d, point_3_4)
        assert isinstance(result, float)

    def test_non_negative(self):
        assert euclidean([5, 3], [1, 0]) >= 0.0

    def test_symmetry(self, origin_2d, point_3_4):
        assert euclidean(origin_2d, point_3_4) == pytest.approx(
            euclidean(point_3_4, origin_2d)
        )

    def test_single_element_vectors(self):
        assert euclidean([3.0], [7.0]) == pytest.approx(4.0)

    def test_float_input(self):
        assert euclidean([0.1, 0.2], [0.4, 0.6]) == pytest.approx(0.5)

    # --- error handling ---

    def test_mismatched_lengths_raises_value_error(self):
        with pytest.raises(ValueError, match="same length"):
            euclidean([1, 2], [1, 2, 3])

    def test_empty_vector_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            euclidean([], [])

    def test_non_numeric_raises_type_error(self):
        with pytest.raises(TypeError, match="numeric"):
            euclidean(["a", "b"], [1, 2])

    def test_2d_array_raises_value_error(self):
        with pytest.raises(ValueError, match="1-D"):
            euclidean([[1, 2], [3, 4]], [[1, 2], [3, 4]])

    def test_nan_raises_value_error(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            euclidean([1.0, float("nan")], [1.0, 2.0])

    def test_inf_raises_value_error(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            euclidean([1.0, float("inf")], [1.0, 2.0])

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            euclidean(None, [1, 2])


# ===========================================================================
# taxicab
# ===========================================================================

class TestTaxicab:
    """Tests for taxicab()."""

    # --- correct values ---

    def test_classic_3_4_grid(self, origin_2d, point_3_4):
        # 3 blocks east + 4 blocks north = 7
        assert taxicab(origin_2d, point_3_4) == pytest.approx(7.0)

    def test_identical_vectors_zero(self):
        assert taxicab([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_unit_vectors(self):
        assert taxicab([1.0, 0.0], [0.0, 1.0]) == pytest.approx(2.0)

    def test_negative_coordinates(self):
        assert taxicab([-3, -4], [0, 0]) == pytest.approx(7.0)

    def test_high_dimensional(self):
        rng = np.random.default_rng(7)
        u = rng.standard_normal(1000)
        v = rng.standard_normal(1000)
        expected = float(np.sum(np.abs(u - v)))
        assert taxicab(u, v) == pytest.approx(expected, rel=1e-10)

    def test_numpy_array_input(self):
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 0.0, 3.0])
        assert taxicab(u, v) == pytest.approx(5.0)

    def test_returns_float(self, origin_2d, point_3_4):
        assert isinstance(taxicab(origin_2d, point_3_4), float)

    def test_non_negative(self):
        assert taxicab([5, 3], [1, 0]) >= 0.0

    def test_symmetry(self, origin_2d, point_3_4):
        assert taxicab(origin_2d, point_3_4) == pytest.approx(
            taxicab(point_3_4, origin_2d)
        )

    def test_single_element_vectors(self):
        assert taxicab([10.0], [3.0]) == pytest.approx(7.0)

    def test_float_input(self):
        # |1.0-1.5| + |2.0-2.5| = 0.5 + 0.5 = 1.0
        assert taxicab([1.0, 2.0], [1.5, 2.5]) == pytest.approx(1.0)

    # --- error handling ---

    def test_mismatched_lengths_raises_value_error(self):
        with pytest.raises(ValueError, match="same length"):
            taxicab([1, 2], [1, 2, 3])

    def test_empty_vector_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            taxicab([], [])

    def test_non_numeric_raises_type_error(self):
        with pytest.raises(TypeError, match="numeric"):
            taxicab(["x"], [1])

    def test_2d_array_raises_value_error(self):
        with pytest.raises(ValueError, match="1-D"):
            taxicab([[1, 2]], [[3, 4]])

    def test_nan_raises_value_error(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            taxicab([float("nan"), 1.0], [0.0, 1.0])

    def test_inf_raises_value_error(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            taxicab([float("inf"), 1.0], [0.0, 1.0])

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            taxicab(None, [1, 2])


# ===========================================================================
# Cross-function properties
# ===========================================================================

class TestDistanceProperties:
    """Metric-axiom and cross-function sanity checks."""

    @pytest.mark.parametrize("fn", [euclidean, taxicab])
    def test_identity_of_indiscernibles(self, fn):
        v = [1.0, 2.0, 3.0]
        assert fn(v, v) == pytest.approx(0.0)

    @pytest.mark.parametrize("fn", [euclidean, taxicab])
    def test_symmetry(self, fn):
        u, v = [1.0, 2.0], [4.0, 6.0]
        assert fn(u, v) == pytest.approx(fn(v, u))

    @pytest.mark.parametrize("fn", [euclidean, taxicab])
    def test_triangle_inequality(self, fn):
        a, b, c = [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]
        assert fn(a, c) <= fn(a, b) + fn(b, c) + 1e-12

    def test_euclidean_le_taxicab(self):
        """Euclidean distance is always <= taxicab distance."""
        rng = np.random.default_rng(99)
        for _ in range(50):
            u = rng.standard_normal(10)
            v = rng.standard_normal(10)
            assert euclidean(u, v) <= taxicab(u, v) + 1e-12
