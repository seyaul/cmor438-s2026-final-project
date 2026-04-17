"""
rice_Ml.supervised_ml.knn
=========================

K-Nearest Neighbours estimators.

Classes
-------
KNNClassifier   — majority-vote classification
KNNRegressor    — mean / inverse-distance-weighted regression
KNNRecommender  — user-based collaborative filtering
"""

from .classifier import KNNClassifier
from .recommender import KNNRecommender
from .regressor import KNNRegressor

__all__ = ["KNNClassifier", "KNNRegressor", "KNNRecommender"]
