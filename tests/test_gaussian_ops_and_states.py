import logging
from math import isclose

from hypothesis import given
from hypothesis import strategies as st
from stellar.cvstates import GaussianState
from stellar.gaussian import GaussianOp, Method, Parameterisation
from stellar.params import GaussianParameters

logger = logging.getLogger(__name__)

logger.info("Starting tests")


@given(st.complex_numbers(min_magnitude=0, max_magnitude=1e6), st.complex_numbers(min_magnitude=0, max_magnitude=1e6))
def test_gauss_op_coherent(op_disp: complex, state_disp: complex) -> None:
    # TODO NEXT don't start from vacuum
    vac = GaussianState(GaussianParameters(x=state_disp.real, y=state_disp.imag, r=1e-5, theta=0))

    op = GaussianOp(
        GaussianParameters(x=op_disp.real, y=op_disp.imag, r=0, theta=0),
        method=Method.recursive,
        param=Parameterisation.Fock,
    )

    output = op @ vac

    # special case with 0 operator squeezing (cosh = 1, sinh = 0)
    assert output.params.x == op_disp.real + state_disp.real
    assert output.params.y == op_disp.imag + state_disp.imag
    assert isclose(output.params.r, 0, abs_tol=1e-5)
    assert isclose(output.params.theta, 0)
