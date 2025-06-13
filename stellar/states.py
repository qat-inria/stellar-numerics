from __future__ import annotations

import dataclasses
import numpy as np
import numpy.typing as npt
from math import isclose
from typing import TypeAlias

Statevector: TypeAlias = npt.NDArray[np.complex128]


@dataclasses.dataclass
class StateFockBasis:
    """Class for single-mode quantum states defines in the Fock basis"""

    statevector: Statevector

    def norm(self) -> float:
        return np.sqrt(np.sum(np.abs(self.statevector) ** 2))

    def is_normalized(self) -> bool:
        return isclose(self.norm(), 1)

    def normalize(self) -> Statevector:
        return self.statevector / self.norm()
