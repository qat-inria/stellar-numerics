import logging

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from stellar.cvstates import CoherentState, CVState, GaussianState, SqueezedVacuumState, StatevectorData, Statevector
from stellar.gaussian import GaussianParameters
from tests.strategies import (
    complex_arrays_st,
)
import cmath
from math import isclose
logger = logging.getLogger(__name__)

logger.info("Starting CV state tests")


@given(complex_arrays_st())
def test_cvstate_init_success(data: StatevectorData) -> None:
    state = CVState(Statevector(data))
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
    cstate = CoherentState(1 + 0.3j)
    sv = cstate.get_statevector(cutoff=10)
    # print(sv.norm, sv)
    assert sv.is_normalized()


def test_cohstate_init_fail() -> None:
    with pytest.raises(TypeError):
        CoherentState("test")  # type: ignore


# Check generic Gaussian state matches coh state. squeezed?


def test_coh_gauss_statevector() -> None:
    # doesn't work with a squeezing phase ... 
    gstate = GaussianState(GaussianParameters(1.0, 0.3, 0.0000001, 0.))
    cohstate = CoherentState(1 + 0.3j)
    cutoff = 10
    print("gaussian")
    print(gstate.get_statevector(cutoff=cutoff).statevector)

    print("coh")
    print(cohstate.get_statevector(cutoff=cutoff).statevector)

    print("fid", np.abs(np.dot(gstate.get_statevector(cutoff=cutoff).statevector.conjugate(), cohstate.get_statevector(cutoff=cutoff).statevector)))
    
    assert isclose(np.abs(np.dot(gstate.get_statevector(cutoff=cutoff).statevector.conjugate(), cohstate.get_statevector(cutoff=cutoff).statevector)), 1, abs_tol=1e-7)
    np.testing.assert_array_almost_equal(gstate.get_statevector(cutoff=cutoff).statevector, cohstate.get_statevector(cutoff=cutoff).statevector, decimal = 2)
    

# use hypothesis over complex numbers and make cutoff large enough!
def test_sqzv_gauss_statevector() -> None:
    # works for real squeezing (r positive, phase 0) cutoff 20
    # problems as soon as there is a phase (r negative doesn't work)
    # not too high squeezing otherwise huge cutoff!
    # doesn't work with a squeezing phase ... 
    sqzamp = 0.5 +0.3j
    gsqzstate = GaussianState(GaussianParameters(0, 0, abs(sqzamp), -cmath.phase(sqzamp)))
    sqzvstate = SqueezedVacuumState(sqzamp)
    cutoff = 20 
    print("gaussian")
    gsqz_statevec = gsqzstate.get_statevector(cutoff=cutoff).statevector
    print(gsqz_statevec)
    sqz_statevec = sqzvstate.get_statevector(cutoff=cutoff).statevector
    print("sqzv")
    print(sqz_statevec)

    print("fid", np.abs(np.sum(gsqz_statevec * sqz_statevec.conjugate()))**2)
    np.testing.assert_array_almost_equal(gsqzstate.get_statevector(cutoff=cutoff).statevector, sqzvstate.get_statevector(cutoff=cutoff).statevector, decimal = 7)
    assert isclose(np.abs(np.sum(gsqzstate.get_statevector(cutoff=cutoff).statevector.conjugate() * sqzvstate.get_statevector(cutoff=cutoff).statevector))**2, 1, abs_tol=1e-5)