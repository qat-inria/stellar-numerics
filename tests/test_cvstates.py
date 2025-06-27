import logging

import pytest
from hypothesis import given
from hypothesis import strategies as st

from stellar.cvstates import CoherentState, CVState, GaussianState, StatevectorData
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


@given(st.complex_numbers(min_magnitude=0, max_magnitude=None))  # technically allows infinity here
def test_cohstate_init_success(amp) -> None:
    cstate = CoherentState(amp)
    # print(cstate.is_gaussian, cstate.params, cstate.amplitude)
    assert isinstance(cstate, GaussianState)
    assert isinstance(cstate, CVState)
    assert isinstance(cstate, CoherentState)
    assert isinstance(cstate.params, GaussianParameters)
    assert isinstance(cstate.amplitude, (float, complex))
    assert cstate.is_gaussian

# @given(st.complex_numbers(min_magnitude=0, max_magnitude=None)) 
def test_cohstate_statevec() -> None:
    cstate = CoherentState(1+.3j)
    sv = cstate.get_statevector(cutoff=10)
    # print(sv.norm, sv)
    assert sv.is_normalized()


def test_cohstate_init_fail() -> None:
    with pytest.raises(TypeError):
        CoherentState("test")