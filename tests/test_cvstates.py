import cmath
import logging
from math import exp, factorial, isclose, sqrt

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from stellar.cvstates import (
    BinomialState,
    CatState,
    CoherentState,
    CompositeStateData,
    CompositeCVState,
    PureCVState,
    DensityMatrix,
    FockState,
    GKPState,
    GaussianState,
    LCGaussianState,
    SqueezedVacuumState,
    Statevector,
    StatevectorData,
    TruncatedParityOp,
)
from stellar.params import GaussianParameters
from tests.strategies import complex_arrays_st, gaussian_parameters_st, tuple_ints_fock_cutoff_st

logger = logging.getLogger(__name__)

logger.info("Starting CV state tests")


@given(complex_arrays_st())
def test_cvstate_init_success(data: StatevectorData) -> None:
    state = PureCVState(Statevector(data))
    assert state.is_gaussian is None
    assert state.statevector is not None


def test_cvstate_init_nodata() -> None:
    state = PureCVState(is_gaussian=True)
    print(state.is_gaussian)
    assert state.is_gaussian


def test_gstate_init_success() -> None:
    gstate = GaussianState(GaussianParameters(1.0, 2.0, 2.5, 0.37))
    print(isinstance(gstate, GaussianState), isinstance(gstate, PureCVState))
    assert isinstance(gstate, GaussianState)
    assert isinstance(gstate, PureCVState)
    assert isinstance(gstate.params, GaussianParameters)
    assert gstate.is_gaussian


@given(st.complex_numbers(min_magnitude=0, max_magnitude=None))  # technically allows infinity here
def test_cohstate_init_success(amp: complex) -> None:
    cstate = CoherentState(amp)
    # print(cstate.is_gaussian, cstate.params, cstate.amplitude)
    assert isinstance(cstate, GaussianState)
    assert isinstance(cstate, PureCVState)
    assert isinstance(cstate, CoherentState)
    assert isinstance(cstate.params, GaussianParameters)
    assert isinstance(cstate.amplitude, (float, complex))
    assert cstate.is_gaussian


# @given(st.complex_numbers(min_magnitude=0, max_magnitude=None))
def test_cohstate_statevec() -> None:
    cstate = CoherentState(1 + 0.3j)
    cutoff = 10
    sv = cstate.get_statevector(cutoff=cutoff)
    # print(sv.norm, sv)
    assert sv.is_normalized()
    assert sv.dim == cutoff + 1


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
    assert sv.dim == cutoff + 1


# tests for density matrices from states
@given(tuple_ints_fock_cutoff_st())
def test_DM_fock(data: tuple[int, int]) -> None:
    n, cutoff = data
    state = FockState(n=n)
    dm = state.get_densitymatrix(cutoff=cutoff)

    assert isinstance(dm, DensityMatrix)
    assert isclose(np.real_if_close(dm.norm), 1)
    assert dm.is_normalized()  # TODO type stuff here
    assert dm.dims == (cutoff + 1,) * 2
    assert isclose(np.real_if_close(dm.purity), 1)
    assert isclose(dm.densitymatrix[n, n].imag, 0)
    assert isclose(np.real_if_close(dm.densitymatrix[n, n]), 1)


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
    cat = CatState(amplitude=amp, parity=parity)

    cat_sv = cat.get_statevector(cutoff=cutoff)

    # assert cat_sv.is_normalized()
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


def test_binomial_state_() -> None:
    """tests:
    binomial(even, 2, 1) = 0.5 |0> + √(3)/2 |4>
    binomial(odd, 2, 1) = √(3)/2 |2> + 1/2 |6>
    binomial(even, 2, 2) = 0.5 |0> + √(3)/2 |6>
    binomial(odd, 1, 2) = |3>
    """

    # TODO refactor and loop?

    st = BinomialState(N=2, S=1, parity=False)

    # TODO define equality on Statevector object directly instead of comparing attributes?

    assert isclose(st.get_statevector(cutoff=6).norm, 1)
    target = np.array([0.5, 0, 0, 0, sqrt(3) / 2, 0, 0])
    np.testing.assert_array_almost_equal(st.get_statevector(cutoff=6).statevector, target)

    st = BinomialState(N=2, S=1, parity=True)
    # print(st.get_statevector(cutoff=6).statevector)
    target = np.array([0, 0, sqrt(3) / 2, 0, 0, 0, 0.5])
    np.testing.assert_array_almost_equal(st.get_statevector(cutoff=6).statevector, target)

    st = BinomialState(N=2, S=2)
    # print(st.get_statevector(cutoff=6).statevector)
    target = np.array([0.5, 0, 0, 0, 0, 0, sqrt(3) / 2])
    np.testing.assert_array_almost_equal(st.get_statevector(cutoff=6).statevector, target)

    st = BinomialState(N=1, S=2, parity=True)
    # print(st.get_statevector(cutoff=6).statevector)
    # max Fock number is 3 but still works.
    target = np.array([0, 0, 0, 1, 0])
    np.testing.assert_array_almost_equal(st.get_statevector(cutoff=4).statevector, target)


def test_GKP_init() -> None:
    # check default
    GKPState()

    # check other parameters too.


def test_init_empty_mixed_state() -> None:
    """test checking that empty mixed states cannot be instantiated."""
    with pytest.raises(ValueError):
        CompositeCVState()


def test_init_ambiguous_mixed_state() -> None:
    """Test checking that mixed states cannot be ambiguously defined."""
    decomp: CompositeStateData = (
        (1.0, PureCVState(Statevector(np.array([0, 1], dtype=np.complex128)))),
        (1.0, PureCVState(Statevector(np.array([2, 1], dtype=np.complex128)))),
    )
    with pytest.raises(ValueError):
        CompositeCVState(matrix=DensityMatrix(np.array([[1, 1], [1, 1]], dtype=np.complex128)), decomposition=decomp)


def test_single_state_decomp() -> None:
    """Check if instantiating with a single pure state results in a warning."""
    decomp: CompositeStateData = ((1.0, PureCVState(Statevector(np.array([0, 1], dtype=np.complex128)))),)
    with pytest.warns():
        CompositeCVState(decomposition=decomp)


def test_get_dm_mixed_vac_decomp_state() -> None:
    #
    """test getting density matrix for mixture of fock states
    with (1/4) |0><0| + (1/6) |2><2| + (7/12) |3><3| using the pure decomposition
    """
    n = 3
    cutoff = n

    decomp: CompositeStateData = ((1 / 4, FockState(n=0)), (1 / 6, FockState(n=n - 1)), (7 / 12, FockState(n=n)))
    st = CompositeCVState(decomposition=decomp)

    dm = st.get_densitymatrix(cutoff=cutoff).densitymatrix

    expected_dm = np.zeros((cutoff + 1,) * 2, dtype=np.complex128)
    # modify in place
    expected_dm[0, 0] = 1 / 4
    expected_dm[n - 1, n - 1] = 1 / 6
    expected_dm[n, n] = 7 / 12

    # print(f"{dm=}\n")
    # print(f"{expected_dm=} \n")

    assert isclose(np.real_if_close(dm.trace()), 1)
    np.testing.assert_array_almost_equal(dm, expected_dm)


def test_get_dm_mixed_vac_mat_state() -> None:
    #
    """test getting density matrix for mixture of fock states
    with (1/4) |0><0| + (1/6) |2><2| + (7/12) |3><3|
    input the matrix directly. Check identity.
    """
    n = 3
    cutoff = n

    expected_dm = np.zeros((cutoff + 1,) * 2, dtype=np.complex128)
    # modify in place
    expected_dm[0, 0] = 1 / 4
    expected_dm[n - 1, n - 1] = 1 / 6
    expected_dm[n, n] = 7 / 12

    st = CompositeCVState(matrix=DensityMatrix(expected_dm))

    dm = st.get_densitymatrix(cutoff=cutoff).densitymatrix

    # print(f"{dm=}\n")
    # print(f"{expected_dm=} \n")

    assert isclose(np.real_if_close(dm.trace()), 1)
    np.testing.assert_array_almost_equal(dm, expected_dm)


@pytest.mark.parametrize("cutoff", range(1, 5))
def test_trunc_parity(cutoff: int) -> None:
    par_op = TruncatedParityOp(cutoff=cutoff)

    dm = par_op.get_densitymatrix(cutoff=cutoff).densitymatrix

    # diag = np.array([(-1)**k for k in range(0, cutoff+1)], dtype=np.complex128)
    expected_dm = np.diag(np.array([(-1) ** k for k in range(0, cutoff + 1)], dtype=np.complex128))

    print(f"{dm=}")
    print(f"{expected_dm=}")
    assert dm.shape == (cutoff + 1,) * 2
    np.testing.assert_array_almost_equal(dm, expected_dm)
