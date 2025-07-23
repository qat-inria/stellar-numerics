"""Module related to the parameterisation of gaussian unitaries
References
[1] Miatto & Quesada https://arxiv.org/abs/2004.11002 (2020)
"""

import logging
from cmath import exp, phase
from enum import Enum, auto
from math import atanh, cosh, sinh, sqrt, tanh

import numpy as np

# just type checking? Nope check fro benchmark but can remove after
from stellar.cvstates import GaussianState, LCGaussianState
from stellar.params import GaussianParameters

# from stellar.cvstates import GaussianState, LCGaussianState

logger = logging.getLogger(__name__)


# target state param
class Parameterisation(Enum):
    Fock = auto()


class Method(Enum):
    direct = auto()
    recursive = auto()


class GaussianOp:
    """Parameterisation of single-mode Gaussians
    we use as in https://arxiv.org/abs/2004.11002 Eq. (43) with no rotation (their phi = 0, their squeezing phase is delta)
    kwargs for left (bra) and right (ket) cutoffs or just optional??
    see also README.MD
    """

    # reuse dataclasse to avoid the init
    # use frozen dataclass to make all attributes read-only (properties)
    # very good!
    # https://docs.python.org/3/library/dataclasses.html#dataclasses.__post_init__
    # use a __post__init__
    # use properties for attributes computed on the fly.
    # need see all cases we can use to modify them

    C: complex
    mean_vector: np.ndarray
    covariance_matrix: np.ndarray

    # TODO rework param and method here here.

    def __init__(self, gauss_param: GaussianParameters, method: Method, param: Parameterisation, **kwargs) -> None:
        self.params = gauss_param

        self.method = method
        if not param == Parameterisation.Fock:
            raise ValueError("Other parameterisation of the target state than Fock is not implemented.")
        self.param = param

        # make all of these properties to compute them on the fly hen needed
        # move outside __init__ to make it work?
        # think with states

        self.alpha = gauss_param.displacement

        # do these need to be exposed?
        match method:
            case Method.recursive:
                # Parameterisation following [1]
                # sech = 1/cosh
                # [1] Eq. (44)
                logger.info("chosen the method recursive")
                self.C = exp(
                    -(
                        abs(self.alpha) ** 2
                        + self.alpha.conjugate() ** 2 * exp(1j * self.params.theta) * tanh(self.params.r)
                    )
                    / 2
                ) / sqrt(cosh(self.params.r))
                logger.debug(f"{self.C=}")
                # [1] Eq. (45)
                self.mean_vector = np.array(
                    [
                        self.alpha.conjugate() * exp(1j * self.params.theta) * tanh(self.params.r) + self.alpha,
                        -self.alpha.conjugate() / cosh(self.params.r),
                    ],
                    dtype=np.complex128,
                )

                # [1] Eq. (46)
                self.covariance_matrix = np.array(
                    [
                        [exp(1j * self.params.theta) * tanh(self.params.r), -1 / cosh(self.params.r)],
                        [-1 / cosh(self.params.r), -exp(-1j * self.params.theta) * tanh(self.params.r)],
                    ],
                    dtype=np.complex128,
                )

            case method.direct:
                pass

    # watch out this creates an import loop
    # # or dynamic dispatch?
    def __matmul__(self, other) -> GaussianState | None:  # GaussianState | LCGaussianState
        # or add type in return
        # how to test that? What is the returned error?
        if not isinstance(other, (GaussianState, LCGaussianState)):
            return NotImplemented  # type: ignore

        if isinstance(other, GaussianState):
            # discard the global phase
            new_disp = (
                self.params.displacement
                + other.params.displacement * cosh(self.params.r)
                - other.params.displacement.conjugate() * exp(1j * self.params.theta) * sinh(self.params.r)
            )

            sqz_expr = (
                exp(1j * self.params.theta) * tanh(self.params.r) + exp(1j * other.params.theta) * tanh(other.params.r)
            ) / (1 + exp(1j * (other.params.theta - self.params.theta)) * tanh(self.params.r) * tanh(other.params.r))

            # print("here", abs(sqz_expr))
            # print("here", atanh(abs(sqz_expr)))
            new_params = GaussianParameters(
                x=new_disp.real, y=new_disp.imag, r=atanh(abs(sqz_expr)), theta=phase(sqz_expr)
            )

            return GaussianState(params=new_params)

        if isinstance(other, LCGaussianState):
            # keep the global phase as it becomes relative (depends on the cases)
            # loop and call itself on GaussianState instances
            return None

        return None

    # should this be here or outside?
    # define a global table of matrix elements with dim cutoff? (equal to size of input state for m (left) and max rank for right (n))
    # add target state somewhere else.
    def build_matrix_fock_basis(
        self, bra_cutoff: int, ket_cutoff: int
    ) -> None:  # or return the matrix and not as attribute?
        # G_{mn} = <m|G|n> Eq 10 Quesada [1]
        # cutoffs are positional arguments so they have to be provided in this order!

        self.matrix_fock_basis = np.zeros((bra_cutoff, ket_cutoff), dtype=np.complex128)

        match self.method:
            case Method.recursive:
                # [1] Eq. (27)
                self.matrix_fock_basis[0, 0] = self.C

                # build first column with [1] Eq (30).
                # no last term
                for m in range(1, bra_cutoff):
                    if m == 1:  # only first term. No root since m == 1
                        self.matrix_fock_basis[m, 0] = self.matrix_fock_basis[m - 1, 0] * self.mean_vector[0]
                    else:  # first two terms
                        self.matrix_fock_basis[m, 0] = (
                            self.matrix_fock_basis[m - 1, 0] * self.mean_vector[0]
                            - sqrt(m - 1) * self.matrix_fock_basis[m - 2, 0] * self.covariance_matrix[0, 0]
                        ) / sqrt(m)

                # build other columns using [1] Eq. (31)
                # column loop, start from second column
                for n in range(1, ket_cutoff):
                    # row loop
                    for m in range(0, bra_cutoff):
                        if n == 1:
                            if m == 0:  # n = 1, m = 0 only first term
                                self.matrix_fock_basis[m, n] = self.matrix_fock_basis[m, n - 1] * self.mean_vector[1]
                            else:  # n = 1, m ≥ 1 no last term
                                self.matrix_fock_basis[m, n] = (
                                    self.matrix_fock_basis[m, n - 1] * self.mean_vector[1]
                                    - sqrt(m) * self.matrix_fock_basis[m - 1, n - 1] * self.covariance_matrix[1, 0]
                                )

                        else:  # generic case n > 1
                            if m == 0:  # n > 1, m = 0 no second term
                                self.matrix_fock_basis[m, n] = (
                                    self.matrix_fock_basis[m, n - 1] * self.mean_vector[1]
                                    - sqrt(n - 1) * self.matrix_fock_basis[m, n - 2] * self.covariance_matrix[1, 1]
                                ) / sqrt(n)
                            else:  # n > 1, m ≥ 1 all three terms
                                self.matrix_fock_basis[m, n] = (
                                    self.matrix_fock_basis[m, n - 1] * self.mean_vector[1]
                                    - sqrt(m) * self.matrix_fock_basis[m - 1, n - 1] * self.covariance_matrix[1, 0]
                                    - sqrt(n - 1) * self.matrix_fock_basis[m, n - 2] * self.covariance_matrix[1, 1]
                                ) / sqrt(n)

            case Method.direct:
                raise NotImplementedError

        return

    # choose specific attributes to compute depending on how the computation will be performed
    # find a way to do it condionally


# @dataclasses.dataclass
# class FockBasisMatrixElement:
#     """compute <m|G|,n>"""

#     param: Parameterisation
#     left_state: int  # m
#     right_state: int  # n

#     def value(self) -> None:  # cmp
#         match self.param:
#             case Parameterisation.direct:
#                 pass
#             case Parameterisation.recursive:
#                 raise NotImplementedError("Feature not yet implemented.")


# To be used both in tests/ and benchmarks/
def check_gaussian_displacement(x: float, y: float) -> None:
    """Eqs. 53 -> 55 of Quesada"""
    gauss_params = GaussianParameters(
        x=x,
        y=y,
        r=0,
        theta=0,
    )
    disp = GaussianOp(gauss_params, method=Method.recursive, param=Parameterisation.Fock)

    assert disp.alpha == x + 1j * y
    assert disp.C == exp(-(abs(disp.alpha) ** 2) / 2)

    target_mean_vector = np.array([disp.alpha, -disp.alpha.conjugate()])
    np.testing.assert_array_equal(disp.mean_vector, target_mean_vector)

    target_cov_matrix = np.array([[0, -1], [-1, 0]])
    np.testing.assert_array_equal(disp.covariance_matrix, target_cov_matrix)
