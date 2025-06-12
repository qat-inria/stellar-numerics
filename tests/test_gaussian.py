import logging
from cmath import exp
from math import cosh, sqrt, tanh, pi

import numpy as np
import pytest
from hypothesis import given, strategies as st

from stellar.gaussian import GaussianOp, Method, Parameterisation


logger = logging.getLogger(__name__)

logger.info("Startng tests")

### NOTE use hypothesis? Yes for random args


@given(st.floats(min_value=-5, max_value=5), st.floats(min_value=-5, max_value=5))
def test_gaussian_displacement(x: float, y: float) -> None:
    """Eqs. 53 -> 55 of Quesada"""
    disp = GaussianOp(x=x, y=y, r=0, theta=0, method=Method.recursive, param=Parameterisation.Fock)

    assert disp.alpha == x + 1j * y
    assert disp.C == exp(-(abs(disp.alpha) ** 2) / 2)

    target_mean_vector = np.array([disp.alpha, -disp.alpha.conjugate()])
    np.testing.assert_array_equal(disp.mean_vector, target_mean_vector)

    target_cov_matrix = np.array([[0, -1], [-1, 0]])
    np.testing.assert_array_equal(disp.covariance_matrix, target_cov_matrix)


@given(st.floats(min_value=-5, max_value=5), st.floats(min_value=0, max_value=2 * pi))
def test_gaussian_squeeze(r: float, theta: float) -> None:
    """Eqs. 47 -> 49 of Quesada"""
    sqz = GaussianOp(x=0, y=0, r=r, theta=theta, method=Method.recursive, param=Parameterisation.Fock)

    assert sqz.C == 1 / sqrt(cosh(sqz.r))

    target_mean_vector = np.array([0, 0])
    np.testing.assert_array_equal(sqz.mean_vector, target_mean_vector)

    target_cov_matrix = np.array(
        [[exp(1j * sqz.theta) * tanh(sqz.r), -1 / cosh(sqz.r)], [-1 / cosh(sqz.r), -exp(-1j * sqz.theta) * tanh(sqz.r)]]
    )
    np.testing.assert_array_equal(sqz.covariance_matrix, target_cov_matrix)


@given(st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=10))
def test_matrix_build(left_cut, right_cut) -> None:
    gauss = GaussianOp(x=0.3, y=0.046, r=1, theta=0.3, method=Method.recursive, param=Parameterisation.Fock)

    gauss.build_matrix(bra_cutoff=left_cut, ket_cutoff=right_cut)

    assert gauss.matrix.shape == (left_cut, right_cut)


# generic test eqs 44-46?
