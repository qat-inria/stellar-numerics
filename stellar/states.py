from __future__ import annotations

from math import isclose
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Statevector: TypeAlias = npt.NDArray[np.complex128]


# all states will derive from this
# what are the shared attributes?
# what are the shared methods??
# THINK
# class CVState():


# move this to methods of CVState class
class StateFockBasis:
    """Class for single-mode quantum states defines in the Fock basis"""

    # assert one-d array

    def __init__(self, statevector: Statevector) -> None:
        if not isinstance(statevector, np.ndarray):
            raise TypeError("Target statevector array has to be oa numpy array.")
        if not statevector.ndim == 1:
            raise ValueError("Target statevector array has to be one dimensional.")

        self.statevector: Statevector = statevector
        self.dim: int = self.statevector.size  # safe since checked that array is 1-dimensional
        # self.norm : float = self.get_norm()

    # use overload? or dispatch?
    # https://stackoverflow.com/questions/6434482/python-function-overloading
    # The @overload decorator from the typing module is used for static type checking, not runtime dispatch. It allows you to hint multiple signatures for a function to type checkers, but you must provide a single implementation:
    # Here, the type checker understands both signatures, but at runtime, only the single implementation is used.

    def __matmul__(self, other: StateFockBasis | Statevector) -> complex | StateFockBasis | Statevector:
        # numpy matmul is inplace or not?
        if isinstance(other, StateFockBasis):
            return np.matmul(self.statevector, other.statevector)
        elif isinstance(other, np.ndarray):
            return np.matmul(self.statevector, other)

    def __repr__(self):
        return f"StateFockBasis({self.statevector})"

    def get_norm(self) -> float:
        # print("norm ", np.sqrt(np.sum(np.abs(self.statevector) ** 2)))
        # print("state ", self.statevector)
        return np.sqrt(np.sum(np.abs(self.statevector) ** 2))

    def is_normalized(self) -> bool:
        return isclose(self.get_norm(), 1)

    def normalize(self) -> None:
        # i n place or not?
        if isclose(self.get_norm(), 0):
            raise ValueError("Cannot normalize the zero vector.")
        self.statevector = self.statevector / self.get_norm()


# write something for Fock states
# more FockState with basis as param
class FockStateFockBasis(StateFockBasis):
    def __init__(self, n: int, cutoff: int):
        # cutoff redund with super dim attribute...# but needed for instantiation

        # cutoff data validation

        if not isinstance(cutoff, int):
            raise TypeError("The Fock space dimension cutoff has to be a integer.")
        if not 0 <= cutoff:
            raise ValueError("The Fock space dimension cutoff has to be positive or 0.")

        # n data validation
        if not isinstance(n, int):
            raise TypeError("The Fock state photon number has to be an integer.")
        if not 0 <= n <= cutoff:
            raise ValueError("The Fock state photon number has to be positive or 0.")

        self.n = n  # Fock state number
        # don't make cutoff an attribute. dim is already there in parent class.

        arr = np.zeros((cutoff,), dtype=np.complex128)
        arr[n] = 1
        super().__init__(arr)
