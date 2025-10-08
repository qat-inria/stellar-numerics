import logging
from cmath import exp
from math import cosh, pi, sqrt, tanh

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from stellar.params import GaussianParameters
from stellar.gaussian import GaussianOp, Method, Parameterisation, check_gaussian_displacement

logger = logging.getLogger(__name__)

logger.info("Starting tests")


def test_gaussian_init_fail_sqz() -> None:
    with pytest.raises(ValueError):
        GaussianParameters(x=0, y=0, r=-5, theta=0.1)


@given(st.floats(min_value=-5, max_value=5), st.floats(min_value=-5, max_value=5))
def test_gaussian_displacement(x: float, y: float) -> None:
    check_gaussian_displacement(x, y)


@given(st.floats(min_value=0, max_value=5), st.floats(min_value=0, max_value=2 * pi))
def test_gaussian_squeeze(r: float, theta: float) -> None:
    """Eqs. 47 -> 49 of Quesada"""
    gauss_params = GaussianParameters(x=0, y=0, r=r, theta=theta)
    sqz = GaussianOp(gauss_params, method=Method.recursive, param=Parameterisation.Fock)

    assert sqz.C == 1 / sqrt(cosh(sqz.params.r))

    target_mean_vector = np.array([0, 0])
    np.testing.assert_array_equal(sqz.mean_vector, target_mean_vector)

    target_cov_matrix = np.array(
        [
            [exp(1j * sqz.params.theta) * tanh(sqz.params.r), -1 / cosh(sqz.params.r)],
            [-1 / cosh(sqz.params.r), -exp(-1j * sqz.params.theta) * tanh(sqz.params.r)],
        ]
    )
    np.testing.assert_array_equal(sqz.covariance_matrix, target_cov_matrix)


@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=10),
    st.floats(min_value=-5, max_value=5),
    st.floats(min_value=-5, max_value=5),
    st.floats(min_value=0, max_value=5),
    st.floats(min_value=0, max_value=2 * pi),
)
def test_matrix_build(left_cut: int, right_cut: int, x: float, y: float, r: float, theta: float) -> None:
    gauss_params = GaussianParameters(x=x, y=y, r=r, theta=theta)
    gauss = GaussianOp(gauss_params, method=Method.recursive, param=Parameterisation.Fock)

    gauss.build_matrix_fock_basis(bra_cutoff=left_cut, ket_cutoff=right_cut)

    assert gauss.matrix_fock_basis.shape == (left_cut + 1, right_cut + 1)


# generic test eqs 44-46?
