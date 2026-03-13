import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from rice_Ml.supervised_ml.Perceptron import perceptron
from rice_Ml.preprocessing import train_test_split, standardize
from rice_Ml.preprocessing import accuracy_score

import matplotlib.pyplot as plt