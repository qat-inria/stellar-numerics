"""Module related to the parameterisation of gaussian unitaries"""

import dataclasses
from cmath import exp
from enum import Enum, auto
import logging
from math import cosh, sqrt, tanh

import numpy as np
from stellar.states import cmp

logger = logging.getLogger(__name__)


# target state param
class Parameterisation(Enum):
    Fock = auto()


class Method(Enum):
    direct = auto()
    recursive = auto()


# don't use dataclass here for conditional attributes
class Gaussian:
    """Parameterisation of single-mode Gaussians
    we use as in https://arxiv.org/abs/2004.11002 Eq. (43) with no rotation (their phi = 0, their squeezing phase is delta)
    kwargs for left (bra) and right (ket) cutoffs or just optional??
    """

    def __init__(
        self, x: float, y: float, r: float, theta: float, method: Method, param: Parameterisation, **kwargs
    ) -> None:
        self.x = x
        self.y = y
        self.r = r
        self.theta = theta
        self.method = method
        if not param == Parameterisation.Fock:
            raise ValueError("Other parameterisation of the target state than Fock is not implemented.")
        self.param = param

        self.alpha = self.x + 1j * self.y

        match method:
            case Method.recursive:
                # Quesada parameterisation
                # sech = 1/cosh
                # Eq. (44)
                logger.info("chosen the method recursive")
                self.C: cmp = exp(
                    -(abs(self.alpha) ** 2 + self.alpha.conjugate() ** 2 * exp(1j * self.theta) * tanh(self.r)) / 2
                ) / sqrt(cosh(self.r))
                logger.debug(f"{self.C=}")
                # Eq. (45)
                # specify data type?? and size??? Might have type pb since rest is complex only not numpy cpx...
                self.mean_vector: np.ndarray = np.array(
                    [
                        self.alpha.conjugate() * exp(1j * self.theta) * tanh(self.r) + self.alpha,
                        -self.alpha.conjugate() / cosh(self.r),
                    ],
                    dtype=np.complex128,
                )

                # Eq. (46)
                self.covariance_matrix: np.ndarray = np.array(
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
    def populate_matrix(self, bra_cutoff: int, ket_cutoff: int) -> cmp:  # G_{mn} = <m|G|n> Eq 10 Quesada
        # cutoffs are positional arguments so they have to be provided in this order!

        self.matrix = np.zeros((bra_cutoff, ket_cutoff), dtype=np.complex128)

        match self.method:
            case Method.recursive:
                # do the recursive computation here.
                self.matrix[0, 0] = self.C
                self.matrix[1, 0] = self.matrix[0, 0] * self.mean_vector[0]
                # build first column
                for m in range(2, bra_cutoff):
                    self.matrix[m, 0] = (self.matrix[m - 1, 0] * self.mean_vector[0] - sqrt(m - 1) * self.matrix[m - 2, 0] * self.covariance_matrix[0, 0]) / sqrt(m)

                pass
            case Method.direct:
                raise NotImplementedError

        return 1.0

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
