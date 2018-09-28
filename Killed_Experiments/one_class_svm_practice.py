from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble   import IsolationForest
from sklearn.neighbors  import LocalOutlierFactor

rng = np.random.RandomState(42)

n_samples           = 200
outliers_fraction   = 0.25
clusters_seperation = [0, 1, 2]

classifier = svm.OneClassSVM(nu = 0.95 * outliers_fraction + 0.05,
                             kernel = "rbf",
                             gamma  = 0.1)
xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
n_inliers  = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)

ground_truth = np.ones(n_samples, dtype = int)
ground_truth[-n_outliers :] = -1 

offset = 0
X1 = 0.3 * np.random.rand(n_inliers // 2, 2) - offset
X2 = 0.3 * np.random.rand(n_inliers // 2, 2) + offset
X  = np.r_[X1, X2]
X  = np.r_[X,  np.random.uniform(low = -6, high = 6, size = (n_outliers, 2))]

classifier.fit(X)
scores_pred = classifier.decision_function(X)
y_pred = classifier.predict(X)
n_errors = (y_pred != ground_truth).sum()