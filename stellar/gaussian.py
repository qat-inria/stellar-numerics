"""Module related to the parameterisation of gaussian unitaries
References
[1] Miatto & Quesada https://arxiv.org/abs/2004.11002 (2020)
"""

import logging
from cmath import exp
from dataclasses import dataclass
from enum import Enum, auto
from math import cosh, sqrt, tanh

import numpy as np

logger = logging.getLogger(__name__)


# target state param
class Parameterisation(Enum):
    Fock = auto()


class Method(Enum):
    direct = auto()
    recursive = auto()


# builtin __init__ and __repr__
@dataclass(frozen=True)  # need frozen to implement __hash__ method
class GaussianParameters:
    """Dataclass containing single-mode Gaussian parameters to be used with both `GaussianStates`and `GaussianOp`.
    NOTE r has to be positive? Or deal separately
        Returns
        -------
        _type_
            _description_
    """

    x: float
    y: float
    r: float
    theta: float

    def __post_init__(self):
        if not isinstance(self.x, (float, int)):
            raise TypeError("Parameter 'x' has to be a float.")
        if not isinstance(self.y, (float, int)):
            raise TypeError("Parameter 'y' has to be a float.")
        if not isinstance(self.r, (float, int)):
            raise TypeError("Parameter 'r' has to be a float.")
        ## NOTE TODO doesn't work so far since need to add constraints in the optimisation algorithm
        # if not self.r >= 0:
        #     raise TypeError("Parameter 'r' has to be a non-negative.")
        if not isinstance(self.theta, (float, int)):
            raise TypeError("Parameter 'theta' has to be a float.")

    @property
    def displacement(self) -> complex:  # TODO: change to displacement
        return self.x + 1j * self.y

    @property
    def squeezing(self) -> complex:
        return self.r * exp(1j * self.theta)


class GaussianOp:
    """Parameterisation of single-mode Gaussians
    we use as in https://arxiv.org/abs/2004.11002 Eq. (43) with no rotation (their phi = 0, their squeezing phase is delta)
    kwargs for left (bra) and right (ket) cutoffs or just optional??
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

    def __init__(self, gauss_param: GaussianParameters, method: Method, param: Parameterisation, **kwargs) -> None:
        self.x = gauss_param.x
        self.y = gauss_param.y
        self.r = gauss_param.r
        self.theta = gauss_param.theta

        self.method = method
        if not param == Parameterisation.Fock:
            raise ValueError("Other parameterisation of the target state than Fock is not implemented.")
        self.param = param

        # make all of these properties to compute them on the fly hen needed
        # move outside __init__ to make it work?
        # think with statesÒ

        self.alpha = gauss_param.displacement

        # do these need to be exposed?
        match method:
            case Method.recursive:
                # Parameterisation following [1]
                # sech = 1/cosh
                # [1] Eq. (44)
                logger.info("chosen the method recursive")
                self.C = exp(
                    -(abs(self.alpha) ** 2 + self.alpha.conjugate() ** 2 * exp(1j * self.theta) * tanh(self.r)) / 2
                ) / sqrt(cosh(self.r))
                logger.debug(f"{self.C=}")
                # [1] Eq. (45)
                self.mean_vector = np.array(
                    [
                        self.alpha.conjugate() * exp(1j * self.theta) * tanh(self.r) + self.alpha,
                        -self.alpha.conjugate() / cosh(self.r),
                    ],
                    dtype=np.complex128,
                )

                # [1] Eq. (46)
                self.covariance_matrix = np.array(
                    [
                        [exp(1j * self.theta) * tanh(self.r), -1 / cosh(self.r)],
                        [-1 / cosh(self.r), -exp(-1j * self.theta) * tanh(self.r)],
                    ],
                    dtype=np.complex128,
                )

            case method.direct:
                pass

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
