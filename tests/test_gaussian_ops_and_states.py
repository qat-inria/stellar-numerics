import logging
from cmath import exp, phase
from math import cosh, isclose, sinh

from hypothesis import given
from hypothesis import strategies as st

from stellar.cvstates import GaussianState
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


# TODO add hypothesis
# commit type annotations and this afterwards
# @given(st.complex_numbers(min_magnitude=0, max_magnitude=1e6), st.complex_numbers(min_magnitude=0, max_magnitude=1e6))
def test_gauss_op_sqz_disp() -> None:
    """check that squeezing a coherent state works as intended"""

    state_disp = -0.3 + 0.7j
    op_sqz = 0.27 - 0.913j

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
    # special case
    # TODO: check equality of GaussianParameters objects instead?

    total_disp = state_disp * cosh(abs(op_sqz)) - state_disp.conjugate() * sinh(abs(op_sqz)) * exp(1j * phase(op_sqz))
    assert output.params.x == total_disp.real
    assert output.params.y == total_disp.imag
    assert isclose(output.params.r, abs(op_sqz), abs_tol=1e-15)
    assert output.params.theta == phase(op_sqz)
