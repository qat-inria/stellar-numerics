import logging
from cmath import exp, phase
from cmath import isclose as cisclose
from math import atanh, cosh, isclose, sinh, sqrt, tanh
from typing import cast

from hypothesis import given
from hypothesis import strategies as st

from stellar.cvstates import GaussianState, LCGaussianState
from stellar.gaussian import GaussianOp, Method, Parameterisation
from stellar.params import GaussianParameters

logger = logging.getLogger(__name__)

logger.info("Starting tests in Gaussian operations and states.")


@given(st.complex_numbers(min_magnitude=0, max_magnitude=1e6), st.complex_numbers(min_magnitude=0, max_magnitude=1e6))
def test_gauss_op_disp_disp(op_disp: complex, state_disp: complex) -> None:
    """check that displacing a coherent state works as intended"""

    # exactly 0 squeezing since not trying to get the statevector
    state = GaussianState(GaussianParameters(x=state_disp.real, y=state_disp.imag, r=0, theta=0))

    op = GaussianOp(
        GaussianParameters(x=op_disp.real, y=op_disp.imag, r=0, theta=0),
        method=Method.recursive,
        param=Parameterisation.Fock,
    )

    output = op @ state

    assert isinstance(output, GaussianState)
    # special case with 0 operator squeezing (cosh = 1, sinh = 0)
    assert output.params.x == op_disp.real + state_disp.real
    assert output.params.y == op_disp.imag + state_disp.imag
    assert output.params.r == 0
    assert output.params.theta == 0


# not too big on the squeezing!
# 50 to match optimization bounds doesn't work.
# go to max 14 with 10^-3 absolute tolerance in modulus tolerance
# errors : math error in the atanh (and cosh) function and numerical precision
# reduce `isclose` precision on modulus
# add minimal and max value to squeezing parameters to avoid problems
# could also put a min value to displacement?
# abs_tol params are the smallest ones allowing the tests to pass witht hese hypothesis parameters.
@given(st.complex_numbers(min_magnitude=1e-6, max_magnitude=14), st.complex_numbers(min_magnitude=0, max_magnitude=1e6))
def test_gauss_op_sqz_disp(op_sqz: complex, state_disp: complex) -> None:
    """check that squeezing a coherent state works as intended"""

    # start from coherent state
    state = GaussianState(GaussianParameters(x=state_disp.real, y=state_disp.imag, r=0, theta=0))
    # and squeeze
    op = GaussianOp(
        GaussianParameters(x=0, y=0, r=abs(op_sqz), theta=phase(op_sqz)),
        method=Method.recursive,
        param=Parameterisation.Fock,
    )

    output = op @ state

    assert isinstance(output, GaussianState)
    # TODO: check equality of GaussianParameters objects instead?

    total_disp = state_disp * cosh(abs(op_sqz)) - state_disp.conjugate() * sinh(abs(op_sqz)) * exp(1j * phase(op_sqz))
    assert isclose(output.params.x, total_disp.real, abs_tol=1e-8)
    assert isclose(output.params.y, total_disp.imag, abs_tol=1e-8)
    assert isclose(output.params.r, abs(op_sqz), abs_tol=1e-3)
    # avoid to check phase equality mod 2π
    assert cisclose(exp(1j * output.params.theta), exp(1j * phase(op_sqz)))


@given(st.complex_numbers(min_magnitude=0, max_magnitude=1e6), st.complex_numbers(min_magnitude=1e-6, max_magnitude=15))
def test_gauss_op_disp_sqz(op_disp: complex, state_sqz: complex) -> None:
    """check that squeezing a coherent state works as intended"""
    # start from squezed vacuum state
    state = GaussianState(GaussianParameters(x=0, y=0, r=abs(state_sqz), theta=phase(state_sqz)))
    # and dispalce
    op = GaussianOp(
        GaussianParameters(x=op_disp.real, y=op_disp.imag, r=0, theta=0),
        method=Method.recursive,
        param=Parameterisation.Fock,
    )

    output = op @ state

    assert isinstance(output, GaussianState)

    assert isclose(output.params.x, op_disp.real)
    assert isclose(output.params.y, op_disp.imag)
    assert isclose(output.params.r, abs(state_sqz), abs_tol=1e-3)
    # avoid to check phase equality mod 2π
    assert cisclose(exp(1j * output.params.theta), exp(1j * phase(state_sqz)))


# keep symmetric bounds
@given(st.complex_numbers(min_magnitude=1e-6, max_magnitude=8), st.complex_numbers(min_magnitude=1e-6, max_magnitude=8))
def test_gauss_op_sqz_sqz(op_sqz: complex, state_sqz: complex) -> None:
    """check that squeezing a coherent state works as intended"""
    # start from squezed vacuum state

    # state_sqz = -0.2 + 0.47j
    # op_sqz = 0.173 - 1.34j
    state = GaussianState(GaussianParameters(x=0, y=0, r=abs(state_sqz), theta=phase(state_sqz)))
    # and displace
    op = GaussianOp(
        GaussianParameters(x=0, y=0, r=abs(op_sqz), theta=phase(op_sqz)),
        method=Method.recursive,
        param=Parameterisation.Fock,
    )

    output = op @ state

    tot_sqz_tanh = (
        exp(1j * op.params.theta) * tanh(op.params.r) + exp(1j * state.params.theta) * tanh(state.params.r)
    ) / (1 + exp(1j * (state.params.theta - op.params.theta)) * tanh(state.params.r) * tanh(op.params.r))
    assert isinstance(output, GaussianState)

    assert isclose(output.params.x, 0)
    assert isclose(output.params.y, 0)
    assert isclose(output.params.r, atanh(abs(tot_sqz_tanh)), abs_tol=1e-3)
    # avoid to check phase equality mod 2π
    assert cisclose(exp(1j * output.params.theta), exp(1j * phase(tot_sqz_tanh)))


@given(st.complex_numbers(min_magnitude=0, max_magnitude=1e6), st.complex_numbers(min_magnitude=0, max_magnitude=1e6))
def test_LCGaussian_op_disp_vac(op_disp: complex, state_disp: complex) -> None:
    """check that displacing a LCGaussian of coherent states works as intended"""

    # cannot do it on length one LCGaussianState
    # allow the behaviour?
    # so do on twice the same state
    g1 = GaussianState(GaussianParameters(x=state_disp.real, y=state_disp.imag, r=0, theta=0))
    g2 = GaussianState(GaussianParameters(x=state_disp.real, y=state_disp.imag, r=0, theta=0))

    state = LCGaussianState(
        (
            (1 / sqrt(2), g1),
            (1 / sqrt(2), g2),
        )
    )

    op = GaussianOp(
        GaussianParameters(x=op_disp.real, y=op_disp.imag, r=0, theta=0),
        method=Method.recursive,
        param=Parameterisation.Fock,
    )

    # unsafe but ok for now
    # TODO remove when typing overload implemented/dynamic dispatch
    output = cast(LCGaussianState, op @ state)

    # print(f"{output=}")

    # just need to check the global phase since matmul on GaussianState has been tested before

    coeff, _ = output.data[0]

    assert isinstance(output, LCGaussianState)

    # no operator squeezing -> only bare displacements in global phase
    assert cisclose(
        coeff,
        exp(
            (
                op.params.displacement * g1.params.displacement.conjugate()
                - op.params.displacement.conjugate() * g1.params.displacement
            )
            / 2
        )
        / sqrt(2),
    )
    # special case with 0 operator squeezing (cosh = 1, sinh = 0)
    # assert output.params.x == op_disp.real + state_disp.real
    # assert output.params.y == op_disp.imag + state_disp.imag
    # assert output.params.r == 0
    # assert output.params.theta == 0
