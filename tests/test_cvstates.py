import logging

from hypothesis import given

from stellar.cvstates import CVState, GaussianState, StatevectorData
from stellar.gaussian import GaussianParameters
from tests.strategies import (
    complex_arrays_st,
)

logger = logging.getLogger(__name__)

logger.info("Starting CV state tests")


@given(complex_arrays_st())
def test_cvstate_init_success(data: StatevectorData) -> None:
    state = CVState(data)
    assert state.is_gaussian is None
    assert state.statevector is not None


def test_cvstate_init_nodata() -> None:
    state = CVState(is_gaussian=True)
    print(state.is_gaussian)
    assert state.is_gaussian


def test_gstate_init_success() -> None:
    gstate = GaussianState(GaussianParameters(1.0, 2.0, 2.5, 0.37))
    print(isinstance(gstate, GaussianState), isinstance(gstate, CVState))
    assert isinstance(gstate, GaussianState)
    assert isinstance(gstate, CVState)
    assert isinstance(gstate.params, GaussianParameters)
    assert gstate.is_gaussian
