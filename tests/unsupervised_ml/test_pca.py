import numpy as np
import pytest
from rice_Ml.unsupervised_ml.decomposition import PCA

def test_pca_explained_variance():
    X = np.random.randn(100, 5)
    pca = PCA(n_components=2)
    pca.fit(X)
    assert pca.components_.shape == (2, 5)
    assert len(pca.explained_variance_) == 2
    assert np.allclose(np.sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_.sum())

def test_pca_transform():
    X = np.random.randn(50, 4)
    pca = PCA(n_components=2)
    X_t = pca.fit_transform(X)
    assert X_t.shape == (50, 2)

def test_pca_inverse():
    X = np.random.randn(30, 3)
    pca = PCA(n_components=2)
    X_t = pca.fit_transform(X)
    X_rec = pca.inverse_transform(X_t)
    assert X_rec.shape == X.shape

def test_pca_whiten():
    X = np.random.randn(20, 6)
    pca = PCA(n_components=3, whiten=True)
    X_t = pca.fit_transform(X)
    # Check unit variance per component
    assert np.allclose(np.var(X_t, axis=0, ddof=1), 1.0, atol=1e-6)