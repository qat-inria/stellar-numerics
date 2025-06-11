import pytest
from stellar.gaussian import Gaussian, Method, Parameterisation
from cmath import exp
from math import sqrt, cosh, tanh
import numpy as np

import logging

logger = logging.getLogger(__name__)

logger.info("Startng tests")


def test_gaussian_displacement() -> None:
    """Eqs. 53 -> 55 of Quesada"""
    disp = Gaussian(x=1, y=1, r=0, theta=0, method=Method.recursive, param=Parameterisation.Fock)

    assert disp.alpha == 1 + 1j
    assert disp.C == exp(-(abs(disp.alpha) ** 2) / 2)

    target_mean_vector = np.array([disp.alpha, -disp.alpha.conjugate()])
    np.testing.assert_array_equal(disp.mean_vector, target_mean_vector)

    target_cov_matrix = np.array([[0, -1], [-1, 0]])
    np.testing.assert_array_equal(disp.covariance_matrix, target_cov_matrix)


def test_gaussian_squeeze() -> None:
    """Eqs. 47 -> 49 of Quesada"""
    sqz = Gaussian(x=0, y=0, r=1, theta=0.3, method=Method.recursive, param=Parameterisation.Fock)

    assert sqz.C == 1 / sqrt(cosh(sqz.r))

    target_mean_vector = np.array([0, 0])
    np.testing.assert_array_equal(sqz.mean_vector, target_mean_vector)

    target_cov_matrix = np.array(
        [[exp(1j * sqz.theta) * tanh(sqz.r), -1 / cosh(sqz.r)], [-1 / cosh(sqz.r), -exp(-1j * sqz.theta) * tanh(sqz.r)]]
    )
    np.testing.assert_array_equal(sqz.covariance_matrix, target_cov_matrix)


# generic test eqs 44-46?
