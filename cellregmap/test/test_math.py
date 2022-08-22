import unittest

from numpy import eye
from numpy.random import RandomState
from numpy_sugar.linalg import economic_qs

from cellregmap._math import (
    # P_matrix,
    QSCov,
    # qmin,
    # rsolve,
    # score_statistic,
    # score_statistic_distr_weights,
    # score_statistic_liu_params,
)

random = RandomState(0)
n_samples = 3
n_covariates = 2

# mean of y
W = random.randn(n_samples, n_covariates) # fixed effect covariates
alpha = array([0.5, -0.2]) # coefficients

# covariance of y
K0 = random.randn(n_samples, n_samples)
K0 = K0 @ K0.T # make it symmetric

v = 0.2
K = v * K0 + eye(n_samples)
dK = K0 # ??

y = random.multivariate_normal(W @ alpha, K)

class TestMath(unittest.TestCase):

    def test_QSCov(self):
        n_samples = K.shape[0]
        QS = economic_qs(K)

