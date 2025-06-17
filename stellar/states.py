from __future__ import annotations

import numpy as np
import numpy.typing as npt
from math import isclose
from typing import TypeAlias

Statevector: TypeAlias = npt.NDArray[np.complex128]



class StateFockBasis:
    """Class for single-mode quantum states defines in the Fock basis"""
    # assert one-d array

    def __init__(self, statevector: Statevector) -> None :
        if not isinstance(statevector, np.ndarray):
            raise TypeError("Target statevector array has to be oa numpy array.")
        if not statevector.ndim == 1:
            raise ValueError("Target statevector array has to be one dimensional.")
        
        self.statevector : Statevector = statevector
        self.dim : int = self.statevector.size # safe since checked that array is 1-dimensional
        # self.norm : float = self.get_norm()

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

    
 
