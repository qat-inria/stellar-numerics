import logging
from math import isclose


from tests.strategies import complex_arrays_st, unit_norm_complex_arrays_st
from hypothesis import given


from stellar.states import StateFockBasis, Statevector

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
    assert isclose(state.get_norm(), 1)


@given(complex_arrays_st())
def test_normalize(data: Statevector) -> None:
    state = StateFockBasis(data)
    state.normalize()
    assert isclose(state.get_norm(), 1, abs_tol=1e-6)
