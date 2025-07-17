import logging
from math import isclose

import numpy as np
from hypothesis import given

from stellar.states import StateFockBasis, Statevector, FockStateFockBasis
from tests.strategies import (
    complex_arrays_st,
    tuple_StateFockBasis_mat_left_st,
    tuple_StateFockBasis_st,
    unit_norm_complex_arrays_st,
    tuple_ints_fock_cutoff_st,
)

logger = logging.getLogger(__name__)

logger.info("Starting state tests")


@given(complex_arrays_st())
def test_init_success(data: Statevector) -> None:
    state = StateFockBasis(data)
    assert state.dim > 0


# failures?


@given(unit_norm_complex_arrays_st())
def test_norm(data: Statevector) -> None:
    state = StateFockBasis(data)
    print(state)
    assert isclose(state.get_norm(), 1, abs_tol=1e-15)


@given(complex_arrays_st())
def test_normalize(data: Statevector) -> None:
    state = StateFockBasis(data)
    print("get norm", state.get_norm())
    state.normalize()
    assert isclose(state.get_norm(), 1, abs_tol=1e-6)


@given(tuple_StateFockBasis_st())
def test_matmul_vectors(stateTuple) -> None:
    state1, state2 = stateTuple
    np.testing.assert_array_equal(state1 @ state2, state1.statevector @ state2.statevector)


# do the same for matrix (left)
@given(tuple_StateFockBasis_mat_left_st())
def test_matmul_vector_mat(data) -> None:
    state, mat = data
    np.testing.assert_array_equal(state @ mat, state.statevector @ mat)
    # assert on shapes?


@given(tuple_ints_fock_cutoff_st())
def test_FockStateFockBasis(data) -> None:
    n, cutoff = data
    state = FockStateFockBasis(n, cutoff)

    print(state.statevector)
    assert np.nonzero(state.statevector) == (np.array([n]),)
    assert isclose(state.get_norm(), 1.0)
