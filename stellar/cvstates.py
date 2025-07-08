"""This modules provides basic definitions and operations on PURE SINGLE-MODE continuous-variable states."""

from __future__ import annotations

import cmath
import functools
from dataclasses import dataclass
from math import cosh, exp, factorial, isclose, sqrt, tanh
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import typing_extensions  # for overriding >+3.12 introduced in typing module
from numpy.polynomial.hermite import hermval

from stellar.gaussian import GaussianParameters

# numpy arrays are homogeneous type wise
StatevectorData: TypeAlias = npt.NDArray[np.complex128] | npt.NDArray[np.float64]


# Single mode pure states only so far.


class Statevector:
    """Class for statevectors in the Fock basis"""

    statevector: StatevectorData

    def __init__(self, data: StatevectorData) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("Target statevector array has to be a numpy array.")
        if not data.ndim == 1:
            raise ValueError(
                "Target statevector array has to be one dimensional. Since the module only supports single-mode pure states."
            )

        self.statevector = data
        self.dim = self.statevector.size  # safe since checked that array is 1-dimensional

    def __repr__(self):
        return f"Statevector({self.statevector})"

    @property
    def norm(self) -> float:
        return np.sqrt(np.sum(np.abs(self.statevector) ** 2))

    # default tolerance is 1e-9. Try 1e-5.
    def is_normalized(self) -> bool:
        return isclose(self.norm, 1, abs_tol=1e-5)

    def normalize(self) -> None:
        # i n place or not?
        if isclose(self.norm, 0):
            raise ValueError("Cannot normalize the zero vector.")
        self.statevector = self.statevector / self.norm


# CVState abstrait puis concret?
@dataclass(frozen=True)
class CVState:
    statevector: Statevector | None = None
    is_gaussian: bool | None = None

    # need same interface for all child classes
    # disregard cutoff
    # all child classes can raise an exception
    # child classes that require a cutoff have to raise a ValueError to eep signature matching
    # inheritance: all method called from A have to be called from B derived from B
    def get_statevector(self, cutoff: int | None = None) -> Statevector:
        if self.statevector is None:
            raise ValueError("No statevector provided.")
        return self.statevector


# Thierry's comments 06-27_2025
# frozen=True gèle aussi les champs hérités, donc le self.is_gaussian dans CVState.__init__ ne pouvait pas fonctionner
# sur un GaussianState

# il vaut mieux ne pas appeler __init__ depuis __post_init__ : en effet, quand __post_init__ est appelé, __init__ a déjà
# été fait une fois, donc tu l'appelles une seconde fois. Ici, c'est bénin, car __init__ ne fait rien de coûteux, mais
# ça ne semble pas être ce qu'on veut ; je pense que le mieux est que CVState soit aussi une dataclass, car c'est aussi
# une simple collection de champs. Une difficulté est que ces champs ont des valeurs par défaut alors que GaussianState
# a un paramètre positionnel obligatoire (sans valeur par défaut) : dans le __init__ généré automatiquement, les
# paramètres des dataclasses hérités vont après les paramètres des dataclasses mères, or il ne peut y avoir de paramètre
# positionnel obligatoire après des paramètres optionnels. La solution est de définir manuellement __init__ dans
# GaussianState.

# Oops, tu as raison : je me serais attendu à ce que le __init__ généré par @dataclass dans la classe héritée appelle le
# __init__ de la classe parent, mais ce n'est pas le cas. Donc __init__ était bien appelé qu'une seule fois.

# Si on en fait une dataclass, ses champs deviennent automatiquement des paramètres dans la méthode __init__ générée, y
# compris dans les classes héritées. Donc le __init__ généré dans GaussianState se trouvait avoir les paramètres dans
# cet ordre : statevector, is_gaussian, params. params ne peut pas ne pas avoir de valeur par défaut si les paramètres
# qui le précède en ont.


# puisque GaussianState est gelé, il faut utiliser object.__setattr__ pour initialiser params.
@dataclass(frozen=True, init=False)  # manually define __init__ for order parameter order reasons
class GaussianState(CVState):
    # one un type pour un champ au niveau de la classe
    # dataclass fait le init en plus
    # mypy voit pas init = False
    params: GaussianParameters  # type: ignore[misc]

    def __init__(self, params: GaussianParameters, statevector: Statevector | None = None):
        super().__init__(statevector=statevector, is_gaussian=True)
        object.__setattr__(self, "params", params)  # since frozen dataclass

    @functools.cache
    @typing_extensions.override
    def get_statevector(self, cutoff: int | None = None) -> Statevector:
        """returns the statevector of a `GaussianState` object. From Chabaud draft Eq. [F15]

        Parameters
        ----------
        cutoff : int
            single-mode Fock space cutoff i.e. the highest Fock number reached

        Returns
        -------
        res : Statevector
            output statevector as a :class:`stellar.cvstates.Statevector` object.

        Raises
        ------
        TypeError
            if the parameter `cutoff`is not an integer.
        TypeError
            if the parameter `cutoff` is not a strictly positive integer.

        Notes
        -----
        Formula has pbs if squeezing goes to zero.
        """
        if not isinstance(cutoff, int):
            raise TypeError("The Fock space cutoff has to be an integer.")
        if not cutoff > 0:
            raise TypeError("The Fock space cutoff has to be greater than zero.")

        thxi = -cmath.exp(-1j * self.params.theta) * tanh(self.params.r)
        hermite_arg = (
            cmath.exp(-1j * self.params.theta / 2)
            * sqrt(tanh(self.params.r) / 2)
            * (self.params.displacement.conjugate() - self.params.displacement / thxi)
        )

        # Create an identity matrix: each row is coefficients for H_n(x)
        coeffs = np.eye(cutoff + 1)

        # Evaluate all Hermite polynomials at z
        hermite_values = np.array([hermval(hermite_arg, c) for c in coeffs])

        data = np.array(
            [
                (-thxi) ** (k / 2)
                / sqrt(2**k * factorial(k) * cosh(self.params.r))
                * cmath.exp(
                    thxi
                    * self.params.displacement.conjugate()
                    * (self.params.displacement.conjugate() - self.params.displacement / thxi)
                    / 2
                )
                * hermite_values[k]
                for k in range(0, cutoff + 1)
            ],
            dtype=np.complex128,
        )

        return Statevector(data)

    # want a get_statevector method for generic state (Zach's ref)
    # but want to be able to overide it in the simplest cases if makes sense


#  or use classmethodclass
# Book:
#     def __init__(self, title, author):
#         self.title = title
#         self.author = author

#     @classmethod
#     def from_string(cls, data_str):
#         title, author = data_str.split(" - ")
#         return cls(title, author)


@dataclass(frozen=True, init=False)  # manually define __init__ to avoid many __init__ calls for differet objects
class CoherentState(GaussianState):  # type: ignore[misc]
    amplitude: complex  # type: ignore[misc]

    def __init__(self, amplitude: complex):
        if isinstance(amplitude, (float, int)):  # type float int are subtypes but not instances
            super().__init__(params=GaussianParameters(float(amplitude), 0, 0, 0))
        elif isinstance(amplitude, complex):
            super().__init__(params=GaussianParameters(amplitude.real, amplitude.imag, 0, 0))
        else:
            raise TypeError("Amplitude parameter has to be a float or number.")
        object.__setattr__(self, "amplitude", amplitude)  # since frozen dataclass

    # try to cache it to avoid recomputing? but maybve don't want to cache it always if  computations are done... store as attribute??
    # need self to be hashable so GaussianParam has to be hashable so frozen dataclass
    @functools.cache
    @typing_extensions.override  # toutes les filles qui implémentent get_statevector to check same signature
    def get_statevector(self, cutoff: int | None = None) -> Statevector:
        """returns the statevector of a `CoherentState` object.

        Parameters
        ----------
        cutoff : int
            single-mode Fock space cutoff i.e. the highest Fock number reached

        Returns
        -------
        res : Statevector
            output statevector as a :class:`stellar.cvstates.Statevector` object.

        Raises
        ------
        TypeError
            if the parameter `cutoff`is not an integer.
        TypeError
            if the parameter `cutoff` is not a strictly positive integer.

        Notes
        -----
        The statevector is computed as

        .. math:: \vert \alpha \rangle = e^{- \abs{\alpha}^2 / 2}\\sum_{n = 0}\frac{\alpha^n}{n!} \vert n \rangle
        """
        if not isinstance(cutoff, int):
            raise TypeError("The Fock space cutoff has to be an integer.")
        if not cutoff > 0:
            raise TypeError("The Fock space cutoff has to be greater than zero.")

        data = exp(-(abs(self.amplitude) ** 2) / 2) * np.array(
            [self.amplitude**n / sqrt(factorial(n)) for n in range(cutoff + 1)]
        )

        return Statevector(data)


@dataclass(frozen=True, init=False)  # manually define __init__ to avoid many __init__ calls for differet objects
class SqueezedVacuumState(GaussianState):  # type: ignore[misc]
    # amplitude = r exp(iθ)
    amplitude: complex  # type: ignore[misc]

    def __init__(self, amplitude: complex):
        if isinstance(amplitude, (complex, float, int)):  # type float int are subtypes but not instances
            super().__init__(params=GaussianParameters(0, 0, abs(amplitude), cmath.phase(amplitude)))
        else:
            raise TypeError("Amplitude parameter has to be a float or number.")
        object.__setattr__(self, "amplitude", amplitude)  # since frozen dataclass

    # try to cache it to avoid recomputing? but maybve don't want to cache it always if  computations are done... store as attribute??
    # need self to be hashable so GaussianParam has to be hashable so frozen dataclass
    @functools.cache
    @typing_extensions.override  # toutes les filles qui implémentent get_statevector to check same signature
    def get_statevector(self, cutoff: int | None = None) -> Statevector:
        """returns the statevector of a `SqueezedVacuumState` object.

        Parameters
        ----------
        cutoff : int
            single-mode Fock space cutoff i.e. the highest Fock number reached

        Returns
        -------
        res : Statevector
            output statevector as a :class:`stellar.cvstates.Statevector` object.

        Raises
        ------
        TypeError
            if the parameter `cutoff`is not an integer.
        TypeError
            if the parameter `cutoff` is not a strictly positive integer.

        Notes
        -----
        The statevector is computed as

        .. math:: \vert \\xi \rangle =
        """
        if not isinstance(cutoff, int):
            raise TypeError("The Fock space cutoff has to be an integer.")
        if not cutoff > 0:
            raise TypeError("The Fock space cutoff has to be greater than zero.")

        # or directly loop over even integers
        data = np.array(
            [
                (-cmath.exp(1j * cmath.phase(self.amplitude)) * tanh(abs(self.amplitude))) ** (n // 2)
                * sqrt(factorial(n))
                / (2 ** (n // 2) * factorial(n // 2))
                if n % 2 == 0
                else 0
                for n in range(cutoff + 1)
            ]
        ) / sqrt(cosh(abs(self.amplitude)))

        return Statevector(data)

    # compute it with min cutoff to satisfy precision requirement?
    # do I want a statevector attribute like other CVState? Guess so. Property?
    # @property
    # def statevector(self) -> float:
    #     return self.getget_statevector


# # move this to methods of CVState class
# class StateFockBasis:
#     """Class for single-mode quantum states defines in the Fock basis"""

#     # assert one-d array

#     def __init__(self, statevector: Statevector) -> None:
#         if not isinstance(statevector, np.ndarray):
#             raise TypeError("Target statevector array has to be a numpy array.")
#         if not statevector.ndim == 1:
#             raise ValueError("Target statevector array has to be one dimensional.")

#         self.statevector: Statevector = statevector
#         self.dim: int = self.statevector.size  # safe since checked that array is 1-dimensional
#         # self.norm : float = self.get_norm()

#     # use overload? or dispatch?
#     # https://stackoverflow.com/questions/6434482/python-function-overloading
#     # The @overload decorator from the typing module is used for static type checking, not runtime dispatch. It allows you to hint multiple signatures for a function to type checkers, but you must provide a single implementation:
#     # Here, the type checker understands both signatures, but at runtime, only the single implementation is used.

#     def __matmul__(self, other: StateFockBasis | Statevector) ->  | StateFockBasis | Statevector:
#         # numpy matmul is inplace or not?
#         if isinstance(other, StateFockBasis):
#             return np.matmul(self.statevector, other.statevector)
#         elif isinstance(other, np.ndarray):
#             return np.matmul(self.statevector, other)

#     def __repr__(self):
#         return f"StateFockBasis({self.statevector})"

#     def get_norm(self) -> float:
#         # print("norm ", np.sqrt(np.sum(np.abs(self.statevector) ** 2)))
#         # print("state ", self.statevector)
#         return np.sqrt(np.sum(np.abs(self.statevector) ** 2))

#     def is_normalized(self) -> bool:
#         return isclose(self.get_norm(), 1)

#     def normalize(self) -> None:
#         # i n place or not?
#         if isclose(self.get_norm(), 0):
#             raise ValueError("Cannot normalize the zero vector.")
#         self.statevector = self.statevector / self.get_norm()


# # write something for Fock states
# # more FockState with basis as param
# class FockStateFockBasis(StateFockBasis):
#     def __init__(self, n: int, cutoff: int):
#         # cutoff redund with super dim attribute...# but needed for instantiation

#         # cutoff data validation

#         if not isinstance(cutoff, int):
#             raise TypeError("The Fock space dimension cutoff has to be a integer.")
#         if not 0 <= cutoff:
#             raise ValueError("The Fock space dimension cutoff has to be positive or 0.")

#         # n data validation
#         if not isinstance(n, int):
#             raise TypeError("The Fock state photon number has to be an integer.")
#         if not 0 <= n <= cutoff:
#             raise ValueError("The Fock state photon number has to be positive or 0.")

#         self.n = n  # Fock state number
#         # don't make cutoff an attribute. dim is already there in parent class.

#         arr = np.zeros((cutoff,), dtype=np.complex128)
#         arr[n] = 1
#         super().__init__(arr)


# class Base:
#     def __init__(self):
#         self.shared_attr = "initial value"

# class ChildA(Base):
#     def __init__(self):
#         super().__init__()

# class ChildB(Base):
#     def __init__(self):
#         super().__init__()

# a = ChildA()
# b = ChildB()
# print(a.shared_attr)  # Output: initial value
# print(b.shared_attr)  # Output: initial value

# class Base:
#     def greet(self):
#         print("Hello from Base")

# class ChildA(Base):
#     def greet(self):
#         print("Hello from ChildA")  # This overrides Base.greet

# class ChildB(Base):
#     pass  # Inherits greet() from Base

# a = ChildA()
# b = ChildB()

# a.greet()  # Output: Hello from ChildA
# b.greet()  # Output: Hello from Base

# class ChildA(Base):
#     def greet(self):
#         super().greet()  # Calls Base.greet()
#         print("...and hello from ChildA")

# This means that instances of the child class are also considered instances of the parent class.
# For example, if we have a parent class Data_Professional and a child class Data_Scientist, an instance of Data_Scientist will be an instance of Data_Professional as well.
