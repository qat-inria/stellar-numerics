import logging
from cmath import exp
from math import cosh, sqrt, tanh, pi

import numpy as np
from hypothesis import given, strategies as st

from stellar.gaussian import GaussianOp, Method, Parameterisation, check_gaussian_displacement


logger = logging.getLogger(__name__)

logger.info("Starting tests")

### NOTE use hypothesis? Yes for random args


@given(st.floats(min_value=-5, max_value=5), st.floats(min_value=-5, max_value=5))
def test_gaussian_displacement(x: float, y: float) -> None:
    check_gaussian_displacement(x, y)


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
