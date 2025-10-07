import cmath
import logging
from math import exp, factorial, isclose, sqrt

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from stellar.cvstates import (
    CatState,
    CoherentState,
    CVState,
    DensityMatrix,
    FockState,
    GaussianState,
    LCGaussianState,
    SqueezedVacuumState,
    Statevector,
    StatevectorData,
)
from stellar.params import GaussianParameters
from tests.strategies import complex_arrays_st, gaussian_parameters_st, tuple_ints_fock_cutoff_st

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
def test_cohstate_init_success(amp: complex) -> None:
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
    # works with a squeezing phase ...
    gstate = GaussianState(GaussianParameters(1.0, 0.3, 0.0000001, 1.5))
    cohstate = CoherentState(1 + 0.3j)
    cutoff = 10
    print("gaussian")
    print(gstate.get_statevector(cutoff=cutoff).statevector)

    print("coh")
    print(cohstate.get_statevector(cutoff=cutoff).statevector)

    print(
        "fid",
        np.abs(
            np.dot(
                gstate.get_statevector(cutoff=cutoff).statevector.conjugate(),
                cohstate.get_statevector(cutoff=cutoff).statevector,
            )
        ),
    )

    assert isclose(
        np.abs(
            np.dot(
                gstate.get_statevector(cutoff=cutoff).statevector.conjugate(),
                cohstate.get_statevector(cutoff=cutoff).statevector,
            )
        ),
        1,
        abs_tol=1e-7,
    )
    np.testing.assert_array_almost_equal(
        gstate.get_statevector(cutoff=cutoff).statevector,
        cohstate.get_statevector(cutoff=cutoff).statevector,
        decimal=7,
    )


# use hypothesis over complex numbers and make cutoff large enough!
def test_sqzv_gauss_statevector() -> None:
    # works for real squeezing (r positive, phase 0) cutoff 20
    # problems as soon as there is a phase (r negative doesn't work)
    # not too high squeezing otherwise huge cutoff!
    # doesn't work with a squeezing phase ...
    sqzamp = 0.5 + 0.3j
    gsqzstate = GaussianState(GaussianParameters(0, 0, abs(sqzamp), cmath.phase(sqzamp)))
    sqzvstate = SqueezedVacuumState(sqzamp)
    cutoff = 20
    # print("gaussian")
    # gsqz_statevec = gsqzstate.get_statevector(cutoff=cutoff).statevector
    # print(gsqz_statevec)
    # sqz_statevec = sqzvstate.get_statevector(cutoff=cutoff).statevector
    # print("sqzv")
    # print(sqz_statevec)
    assert sqzvstate.is_gaussian
    # print("fid", np.abs(np.sum(gsqz_statevec * sqz_statevec.conjugate())) ** 2)
    np.testing.assert_array_almost_equal(
        gsqzstate.get_statevector(cutoff=cutoff).statevector,
        sqzvstate.get_statevector(cutoff=cutoff).statevector,
        decimal=7,
    )
    assert isclose(
        np.abs(
            np.sum(
                gsqzstate.get_statevector(cutoff=cutoff).statevector.conjugate()
                * sqzvstate.get_statevector(cutoff=cutoff).statevector
            )
        )
        ** 2,
        1,
        abs_tol=1e-5,
    )


@given(st.integers(min_value=1))
def test_Fock(n: int) -> None:
    state0 = FockState(0)
    state = FockState(n=n)

    assert state0.is_gaussian
    assert not state.is_gaussian


# tests for density matrices from states
@given(tuple_ints_fock_cutoff_st())
def test_SV_fock(data: tuple[int, int]) -> None:
    n, cutoff = data
    print("n", n, "cutoff", cutoff)
    state = FockState(n=n)

    sv = state.get_statevector(cutoff=cutoff)

    print(sv.statevector)

    assert isclose(sv.norm, 1)


# tests for density matrices from states
@given(tuple_ints_fock_cutoff_st())
def test_DM_fock(data: tuple[int, int]) -> None:
    n, cutoff = data
    print("n", n, "cutoff", cutoff)
    state = FockState(n=n)
    dm = state.get_densitymatrix(cutoff=cutoff)
    print(f"{dm=}")
    assert isinstance(dm, DensityMatrix)
    assert isclose(np.real_if_close(dm.norm), 1)
    assert dm.is_normalized()  # TODO type stuff here
    assert dm.dims == (cutoff,) * 2
    assert isclose(np.real_if_close(dm.purity), 1)
    assert isclose(dm.densitymatrix[n, n].imag, 0)
    assert isclose(np.real_if_close(dm.densitymatrix[n, n]), 1)


# test DensityMatrix


# tests LCGaussian
# only one state
@given(gaussian_parameters_st())
def test_LCGaussian_init_false_one(params: GaussianParameters) -> None:
    with pytest.raises(ValueError):
        LCGaussianState(((1, GaussianState(params=params)),))


# more than length one and non gaussian states
@given(st.integers(min_value=1))
def test_LCGaussian_init_false_ngauss(n: int) -> None:
    # need at least 2 non-gaussian to catch that error
    # otherwise get length-1 error
    with pytest.raises(TypeError):
        LCGaussianState(
            (
                (1, FockState(n)),
                (1, FockState(n + 1)),
            )  # type: ignore
        )


# @given(gaussian_parameters_st(), gaussian_parameters_st())
# def test_LCGAussian_init_false_nnormed(params1, params2) -> None:
#     # need at least 2
#     with pytest.raises(ValueError):
#         LCGaussianState([(1, GaussianState(params=params1)), (1, GaussianState(params=params2))])


# success
@given(gaussian_parameters_st(), gaussian_parameters_st())
def test_LCGaussian_success(params1: GaussianParameters, params2: GaussianParameters) -> None:
    g1 = GaussianState(params=params1)
    g2 = GaussianState(params=params2)

    LCGaussianState(
        (
            (1 / sqrt(2), g1),
            (1 / sqrt(2), g2),
        )
    )


# success statevec
@given(gaussian_parameters_st(), gaussian_parameters_st())
def test_LCGaussian_statevec(params1: GaussianParameters, params2: GaussianParameters) -> None:
    g1 = GaussianState(params=params1)
    g2 = GaussianState(params=params2)

    LCGaussianState(
        (
            (1 / sqrt(2), g1),
            (1 / sqrt(2), g2),
        )
    )

    # TODO test statevec construction


# success cat state
# @given(gaussian_parameters_st(), gaussian_parameters_st())
# amp cutoff
@given(st.complex_numbers(allow_nan=False, allow_infinity=False, min_magnitude=1e-5, max_magnitude=1e5), st.booleans())
def test_cat_state_init_success(amp: complex, parity: bool) -> None:
    cat = CatState(amplitude=amp, parity=parity)

    assert not cat.is_gaussian
    assert cat.amplitude == amp
    assert cat.parity == parity

    norm = sqrt(2 * (1 + +((-1) ** parity) * exp(-2 * abs(amp) ** 2)))

    assert cat.data == (
        (1.0 / norm, CoherentState(amp)),
        ((-1) ** parity / norm, CoherentState(-amp)),
    )


@given(
    st.complex_numbers(allow_nan=False, allow_infinity=False, min_magnitude=1e-5, max_magnitude=1e5),
    st.booleans(),
    st.integers(min_value=1, max_value=20),
)
def test_cat_plus_state_statevec(amp: complex, parity: bool, cutoff: int) -> None:
    # amp = 0.5 + 0.3j
    # cutoff = 5
    cat_plus = CatState(amplitude=amp, parity=parity)

    cat_sv = cat_plus.get_statevector(cutoff=cutoff)
    # print(f"{cat_plus=}")
    target = (
        exp(-(abs(amp) ** 2) / 2)
        * np.array(
            [2 * amp**k / sqrt(factorial(k)) if k % 2 == int(parity) else 0 for k in range(0, cutoff + 1)],
            dtype=np.complex128,
        )
        / sqrt(2 * (1 + +((-1) ** parity) * exp(-2 * abs(amp) ** 2)))
    )

    np.testing.assert_array_almost_equal(cat_sv.statevector, target)
