import dataclasses
from collections.abc import Sequence
from math import isclose, sqrt

# alias for complex numbers
# python ≥ 3.12 only
# type cmp = complex | float
# else
from typing import TypeAlias

cmp: TypeAlias = complex | float


@dataclasses.dataclass
class StateFockBasis:
    """Class for single-mode quantum states defines in the Fock basis"""

    statevector: Sequence[cmp]  # see https://docs.python.org/3/library/collections.abc.html

    def norm(self) -> float:
        return sqrt(sum(abs(ell) ** 2 for ell in self.statevector))

    def normalize(self) -> float:
        if not isclose(self.norm, 1):
            print("initial state not normalized. Normalizing....")
        # division not defined for sequences -> numpy arrays
        return self.statevector / self.norm()
