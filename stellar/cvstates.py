"""This modules provides basic definitions and operations on PURE SINGLE-MODE continuous-variable states."""

from __future__ import annotations

import cmath
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil, comb, cosh, exp, factorial, isclose, log, sqrt, tanh
from math import pi as π
from typing import Generic, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import typing_extensions  # for overriding >+3.12 introduced in typing module
from numpy.polynomial.hermite import hermval

from stellar.params import GaussianParameters

# numpy arrays are homogeneous type wise
# no knowledge of the number of dimensions
# so works for both DensityMatrix and Statevector
StatevectorData: TypeAlias = npt.NDArray[np.complex128] | npt.NDArray[np.float64]

# make statevector a property of the state??
# Single mode pure states only so far.


# make it inherit from numpy arrays to void redefining everything
# (like conj and so on...) but makes things fail (np.ndarray) ...
class Statevector:
    """A single-mode statevector represented in the Fock basis.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional array of Fock-basis amplitudes. The array is retained
        by reference.

    Raises
    ------
    TypeError
        If ``data`` is not a NumPy array.
    ValueError
        If ``data`` is not one-dimensional.
    """

    statevector: StatevectorData

    def __init__(self, data: StatevectorData) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("Target statevector array has to be a numpy array.")
        if not data.ndim == 1:
            raise ValueError("Target statevector array has to be one dimensional.")

        self.statevector = data
        self.dim = self.statevector.size  # safe since checked that array is 1-dimensional

    def __repr__(self):
        return f"Statevector({self.statevector})"

    @property
    def norm(self) -> float:
        """float: Euclidean norm of the statevector."""
        return np.sqrt(np.sum(np.abs(self.statevector) ** 2))

    # default tolerance is 1e-9. Try 1e-5.
    def is_normalized(self) -> bool:
        """Return whether the statevector has unit norm.

        Returns
        -------
        bool
            ``True`` when the norm is within an absolute tolerance of ``1e-5``
            of one.
        """
        return isclose(self.norm, 1, abs_tol=1e-5)

    def normalize(self) -> None:
        """Normalize this statevector in place.

        Raises
        ------
        ValueError
            If the statevector is the zero vector.
        """
        # i n place or not?
        if isclose(self.norm, 0):
            raise ValueError("Cannot normalize the zero vector.")
        self.statevector = self.statevector / self.norm


class Matrix:
    """A square matrix, such as a density matrix or operator, in the Fock basis.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional square array. The array is retained by reference.

    Raises
    ------
    TypeError
        If ``data`` is not a NumPy array.
    ValueError
        If ``data`` is not a square, two-dimensional array.
    """

    # same type as statevector since no way to type annotate number of dimensions
    matrix: StatevectorData

    def __init__(self, data: StatevectorData) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("Target matrix array has to be a numpy array.")
        if not data.ndim == 2:
            raise ValueError("Target matrix array has to be a matrix.")
        if not data.shape[0] == data.shape[1]:
            raise ValueError("Target matrix array has to be a square matrix.")

        # check psdness and hermiticity?

        self.matrix = data
        self.dims = self.matrix.shape  # safe since checked that array is 1-dimensional

    def __repr__(self):
        return f"DensityMatrix({self.matrix})"

    @property
    def norm(self) -> float | complex:
        """float or complex: Trace of the matrix."""
        return np.trace(self.matrix)

    @property
    def purity(self) -> float:
        """float: Trace of the squared, unit-trace matrix.

        Notes
        -----
        If the matrix is not normalized, this property normalizes it in place
        before computing the purity.
        """
        if not self.is_normalized():
            self.normalize()
        return np.trace(self.matrix @ self.matrix)

    # default tolerance is 1e-9. Try 1e-5.
    def is_normalized(self) -> bool:
        """Return whether the matrix has unit trace.

        Returns
        -------
        bool
            ``True`` when the real trace is within an absolute tolerance of
            ``1e-5`` of one.

        Raises
        ------
        ValueError
            If the trace has a nonzero imaginary part.
        """
        if not isclose(self.norm.imag, 0):
            raise ValueError("Norm cannot be cast to real so the density matrix is probably not hermitian.")
        return isclose(np.real_if_close(self.norm), 1, abs_tol=1e-5)

    def normalize(self) -> None:
        """Normalize this matrix to unit trace in place.

        Raises
        ------
        ValueError
            If the matrix has zero trace.
        """
        if np.isclose(self.norm, 0):
            raise ValueError("Cannot normalize the zero matrix.")
        self.matrix = (self.matrix / self.norm).astype(
            np.complex128
        )  # type de scalaire pas tableau. pas infer type complex128/float ou cpx


# CVState abstrait puis concret?
@dataclass(frozen=True)
class PureCVState:
    """Base representation of a pure, single-mode continuous-variable state.

    Parameters
    ----------
    statevector : Statevector or None, optional
        Explicit Fock-basis representation, when already available.
    is_gaussian : bool or None, optional
        Whether the state is Gaussian, if known.
    """

    statevector: Statevector | None = None
    is_gaussian: bool | None = None

    # need same interface for all child classes
    # disregard cutoff
    # all child classes can raise an exception
    # child classes that require a cutoff have to raise a ValueError to keep signature matching
    # inheritance: all method called from A have to be called from B, derived from B
    def get_statevector(self, cutoff: int | None = None) -> Statevector:
        """Return the stored statevector.

        Parameters
        ----------
        cutoff : int or None, optional
            Accepted for compatibility with state subclasses; unused by this
            base implementation.

        Returns
        -------
        Statevector
            Stored Fock-basis representation.

        Raises
        ------
        ValueError
            If this instance does not contain a statevector.
        """
        if self.statevector is None:
            raise ValueError("No statevector provided.")
        # self.is_gaussian = cast(bool, 42) # clairement faux mais existe cas où l'inferrence ne marche pas. Eviter types trop généraux, réduire et mettre cast où on peut prouver le type.
        return self.statevector

    # since all states have get_statevector method
    # this should be fine
    def get_densitymatrix(self, cutoff: int | None = None) -> Matrix:
        """Construct the density matrix ``|psi><psi|`` for this state.

        Parameters
        ----------
        cutoff : int or None, optional
            Cutoff passed to :meth:`get_statevector`.

        Returns
        -------
        Matrix
            Outer product of the statevector with its complex conjugate.
        """
        # NOTE this should work?
        # if self.statevector is None:
        #     raise ValueError("No statevector provided.")
        return Matrix(
            np.outer(
                self.get_statevector(cutoff=cutoff).statevector,
                self.get_statevector(cutoff=cutoff).statevector.conjugate(),
            ).astype(np.complex128)  # pas un cast. outer ne guarantie pas la précision du complexe mais type oui
        )  # typing.cast python : impose au typeur de croire que 'est d'un type donné
        # TODO type stuff here

    def __repr__(self) -> str:
        return f"PureCVState(statevector={self.statevector}, is_gaussian={self.is_gaussian})"


# TODO composite state hierarchy and abstract classes to distinguish witnesses from DM
# get_DM -> get_operator return get_DM for mixed states
# want init dm or list

PureDecompositionData: TypeAlias = tuple[tuple[float | int, PureCVState], ...]

# need immutable (or frozenset/abstractset)


# init directly by matrix or pure state decomposition (not check unit trace and psdness)
# not that for statevectors, the get_statevec method doesn't modify in place the field but is functional so don't do it here either
# TODO: when inputting data directly, do it as arrays, not custom objects, it's cumbersome
# TODO for the instantiation (decomp vs matrix) use an Enum to exhaust all cases and disjunction
# TODO change to HermitianCVOperator
T = TypeVar("T", Matrix, PureDecompositionData)


@dataclass(frozen=True)
class HermitianCVOp(Generic[T]):
    """A Hermitian operator represented by a matrix or pure-state decomposition.

    Parameters
    ----------
    data : Matrix or tuple of (float, PureCVState)
        Either a Fock-basis matrix or a sequence of real-weighted pure-state
        terms. The decomposition may represent an operator that is not a
        normalized density matrix.

    Warns
    -----
    UserWarning
        If the decomposition contains a single pure state.
    """

    data: T  # Matrix | PureDecompositionData

    def __post_init__(self) -> None:
        if (
            not isinstance(self.data, Matrix) and len(self.data) == 1
        ):  # if first predicate is true second makes sense (can be checked)
            # cannot be none here none type ignore is safe
            warnings.warn("A composite state with a single pure state in its decomposition is just a pure state.")

    def get_densitymatrix(self, cutoff: int | None = None) -> Matrix:
        """Return the matrix representation of this operator.

        Parameters
        ----------
        cutoff : int or None, optional
            Fock cutoff passed to component states when ``data`` is a
            decomposition.

        Returns
        -------
        Matrix
            The stored matrix or the matrix assembled from the decomposition.
        """
        if not isinstance(self.data, Matrix):
            # compute from decomposition
            # apparently need explicit list conversion for sum to work
            data = np.sum(
                [
                    coeff
                    * np.outer(
                        state.get_statevector(cutoff=cutoff).statevector,
                        state.get_statevector(cutoff=cutoff).statevector.conjugate(),
                    ).astype(np.complex128)
                    for coeff, state in self.data
                ],
                axis=0,
            )

            return Matrix(data)

        else:
            return self.data

    # TODO think about it since not sure to have a decomposition
    def __iter__(self) -> Iterator[tuple[float | int, PureCVState]]:
        """Iterate over decomposition terms.

        Yields
        ------
        tuple of (float or int, PureCVState)
            A real coefficient and its pure state.

        Raises
        ------
        TypeError
            If this operator is stored as a matrix rather than a decomposition.
        """
        if not isinstance(self.data, Matrix):
            return iter(self.data)
        else:
            raise TypeError("Cannot iterate on the object.")

    def __repr__(self) -> str:
        return f"HermitianCVOp(data={self.data})"


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
class GaussianState(PureCVState):
    """A pure single-mode Gaussian state.

    Parameters
    ----------
    params : GaussianParameters
        Displacement and squeezing parameters.
    statevector : Statevector or None, optional
        An optional explicit Fock-basis representation.
    """

    # one un type pour un champ au niveau de la classe
    # dataclass fait le init en plus
    # mypy voit pas init = False
    params: GaussianParameters  # type: ignore[misc]

    # TODO can allow input gaussian by statevec or not? Not sure it really makes sense...
    def __init__(self, params: GaussianParameters, statevector: Statevector | None = None):
        super().__init__(statevector=statevector, is_gaussian=True)
        object.__setattr__(self, "params", params)  # since frozen dataclass

    # @functools.cache
    @typing_extensions.override
    def get_statevector(self, cutoff: int | None = None) -> Statevector:  # signature has to match since override
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
        Formula is ill-defined for zero squeezing.
        Use a very small squeezing amplitude or :class:`stellar.cvstates.CoherentState` instead.
        Need to shift convention with respect to ours : xi-> xi* so theta -> - theta wrt paper
        """
        if not isinstance(cutoff, int):
            raise TypeError("The Fock space cutoff has to be an integer.")
        if not cutoff >= 0:  # NOTE useless now?
            raise TypeError("The Fock space cutoff has to be greater or equal than zero.")

        thxi = -cmath.exp(1j * self.params.theta) * tanh(self.params.r)
        hermite_arg = (
            cmath.exp(1j * self.params.theta / 2)
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
                for k in range(cutoff + 1)
            ],
            dtype=np.complex128,
        )

        return Statevector(data)

    def __repr__(self) -> str:
        return f"GaussianState(params={self.params})"

    # overload __matmul__(self, other) for gaussian operations
    # if type(other) not implemented
    # return NotImplemented -> appel rmatmul -> echoue aussi (operateur non defini)
    # a @ b = a.matmul(b) if fail b.rmatmul

    # want a get_statevector method for generic state (Zach's ref)
    # but want to be able to overide it in the simplest cases if makes sense


@dataclass(frozen=True, init=False)  # manually define __init__ for order parameter order reasons
class FockState(PureCVState):
    """A number state ``|n>`` in the Fock basis.

    Parameters
    ----------
    n : int
        Non-negative Fock number.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    """

    n: int  # type: ignore

    def __init__(self, n: int):
        """Initialize a Fock number state.

        Parameters
        ----------
        n : int
            Non-negative Fock number.

        Raises
        ------
        TypeError
            If ``n`` is not an integer.
        """
        if not isinstance(n, int):
            raise TypeError("The Fock number has to be an integer.")
        if n == 0:
            super().__init__(statevector=None, is_gaussian=True)
        else:
            super().__init__(statevector=None, is_gaussian=False)
        object.__setattr__(self, "n", n)  # since frozen dataclass

    # @functools.cache
    @typing_extensions.override
    def get_statevector(self, cutoff: int | None = None) -> Statevector:
        """returns the statevector of a `FockState` object.

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
        """
        if not isinstance(cutoff, int):
            raise TypeError("The Fock space cutoff has to be an integer.")
        if not cutoff >= self.n >= 0:
            raise ValueError(
                "The Fock space cutoff has to be greater than zero and larger or equal than the Fock number."
            )

        # When you declare a NumPy array, you declare it with a type,
        # and if you append anything to that array, it will be converted to that type
        # Numpy arrays are homogeneous
        data = np.zeros(cutoff + 1, dtype=np.complex128)
        data[self.n] = 1

        return Statevector(data)

    def __repr__(self) -> str:
        return f"FockState(n={self.n})"


@dataclass(frozen=True, init=False)  # manually define __init__ to avoid many __init__ calls for differet objects
class CoherentState(GaussianState):  # type: ignore[misc]
    """A single-mode coherent state.

    Parameters
    ----------
    amplitude : complex
        Coherent-state amplitude. Real values are interpreted as amplitudes
        with zero imaginary part.

    Raises
    ------
    TypeError
        If ``amplitude`` is not a real or complex number.
    """

    amplitude: complex  # type: ignore[misc]

    def __init__(self, amplitude: complex):
        """Initialize a coherent state from its complex amplitude.

        Parameters
        ----------
        amplitude : complex
            Displacement amplitude; a real number is also accepted.

        Raises
        ------
        TypeError
            If ``amplitude`` is not a real or complex number.
        """
        if isinstance(amplitude, (float, int)):  # type float int are subtypes but not instances
            super().__init__(params=GaussianParameters(float(amplitude), 0, 0, 0))
        elif isinstance(amplitude, complex):
            super().__init__(params=GaussianParameters(amplitude.real, amplitude.imag, 0, 0))
        else:
            raise TypeError("Amplitude parameter has to be a float or number.")
        object.__setattr__(self, "amplitude", amplitude)  # since frozen dataclass

    # try to cache it to avoid recomputing? but maybve don't want to cache it always if  computations are done... store as attribute??
    # need self to be hashable so GaussianParam has to be hashable so frozen dataclass
    # mutable objects are not hashable!
    # @functools.cache
    @typing_extensions.override  # toutes les filles qui implémentent get_statevector to check same signature
    def get_statevector(self, cutoff: int | None = None) -> Statevector:
        r"""returns the statevector of a `CoherentState` object.

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

        .. math:: \vert \alpha \rangle = e^{- \vert\alpha\vert^2 / 2}\\sum_{n = 0}\frac{\alpha^n}{n!} \vert n \rangle
        """
        if not isinstance(cutoff, int):
            raise TypeError("The Fock space cutoff has to be an integer.")
        if not cutoff > 0:
            raise TypeError("The Fock space cutoff has to be greater than zero.")

        data = exp(-(abs(self.amplitude) ** 2) / 2) * np.array(
            [self.amplitude**n / sqrt(factorial(n)) for n in range(cutoff + 1)]
        )

        return Statevector(data)

    def __repr__(self) -> str:
        return f"CoherentState(amplitude={self.amplitude})"


@dataclass(frozen=True, init=False)
class SqueezedVacuumState(GaussianState):  # type: ignore[misc]
    """A single-mode squeezed vacuum state.

    Parameters
    ----------
    amplitude : complex
        Complex squeezing parameter. Its magnitude is the squeezing strength
        and its phase is the squeezing angle.

    Raises
    ------
    TypeError
        If ``amplitude`` is not a real or complex number.
    """

    # amplitude = r exp(iθ)
    amplitude: complex  # type: ignore[misc]

    def __init__(self, amplitude: complex):
        """Initialize a squeezed vacuum from its complex squeezing parameter.

        Parameters
        ----------
        amplitude : complex
            Squeezing parameter; a real number is also accepted.

        Raises
        ------
        TypeError
            If ``amplitude`` is not a real or complex number.
        """
        if isinstance(amplitude, (complex, float, int)):  # type float int are subtypes but not instances
            super().__init__(params=GaussianParameters(0, 0, abs(amplitude), cmath.phase(amplitude)))
        else:
            raise TypeError("Amplitude parameter has to be a float or number.")
        object.__setattr__(self, "amplitude", amplitude)  # since frozen dataclass

    # try to cache it to avoid recomputing? but maybve don't want to cache it always if  computations are done... store as attribute??
    # need self to be hashable so GaussianParam has to be hashable so frozen dataclass
    # @functools.cache
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
            if the parameter `cutoff` is not an integer.
        ValueError
            if the parameter `cutoff` is not a strictly positive integer.

        Notes
        -----
        The statevector is computed as

        .. math:: \vert \\xi \rangle =
        """
        if not isinstance(cutoff, int):
            raise TypeError("The Fock space cutoff has to be an integer.")
        if not cutoff > 0:
            raise ValueError("The Fock space cutoff has to be greater than zero.")

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

    def __repr__(self) -> str:
        return f"SqueezedVacuumState(amplitude={self.amplitude})"


# need to make that a frozen dataclass for same hashing issues
# Sequence pas hashable car list mutable
# or Mapping instead of tuple?
# tuple of (coeff, GaussianState) pairs
LCGaussianData: TypeAlias = tuple[tuple[complex, GaussianState], ...]  # need immutable (or frozenset/abstractset)


# rego to type alias
# @dataclass(frozen=True)
# class LCGaussianData:
#     # init automatique est magique (fait des setattr)
#     # sequence is mutable
#     data: tuple[tuple[complex, CVState], ...]  # otherwise only 1 element and second branch always false

#     def __post_init__(self):
#         if len(self.data) == 1:
#             raise ValueError(
#                 "A linear combination of a single Gaussian state is a Gaussian state, so use a `GaussianState`object instead."
#             )
#         # TODO tune so that FockState(0) is ok. Or use GaussianState with (0,) * 4 params
#         # but ill defined statevec-wise I guess. Deal with that case.
#         if not all(
#             [isinstance(state, GaussianState) for _, state in self.data]
#         ):  # Fock 0 should be included (both Gaussian and Fock)
#             raise TypeError("All states in a LCGaussianState have to be GaussianState objects.")

#         # object.__setattr__(self, "data", self.data)

#     def __iter__(self) -> Iterator[tuple[complex, CVState]]:
#         return iter(self.data)


@dataclass(frozen=True, init=False)
class LCGaussianState(PureCVState):
    """A finite linear combination of single-mode Gaussian states.

    Parameters
    ----------
    data : tuple of (complex, GaussianState)
        Coefficient-state pairs defining the superposition. A one-term input is
        rejected; the coefficients are not automatically normalized.

    Raises
    ------
    ValueError
        If ``data`` contains exactly one term.
    TypeError
        If any term is not a :class:`GaussianState`.
    """

    data: LCGaussianData  # type: ignore

    def __init__(self, data: LCGaussianData) -> None:
        """Initialize a Gaussian-state linear combination.

        Parameters
        ----------
        data : tuple of (complex, GaussianState)
            Coefficient-state pairs. The tuple must contain at least two
            Gaussian states.

        Raises
        ------
        ValueError
            If ``data`` has exactly one term.
        TypeError
            If a term's state is not a :class:`GaussianState`.
        """
        # default to non Gaussian
        # it is not if two states are the same in a length 2 sequence
        # redifine equality for GaussianStates == equality of GaussianParameters
        # define equality of GaussianParameters
        # skip that for now

        # coefficients have to be normalized
        # extract coeff and state lists separately?
        super().__init__(statevector=None, is_gaussian=False)
        object.__setattr__(self, "data", data)

        # TODO add empty tuple input handling?
        # NOTE actually this could be useful to handle global phases in single gausian states...
        if len(self.data) == 1:
            raise ValueError(
                "A linear combination of a single Gaussian state is a Gaussian state, so use a `GaussianState` object instead."
            )

        # TODO tune so that FockState(0) is ok. Or use GaussianState with (0,) * 4 params
        # but ill defined statevec-wise I guess. Deal with that case.
        if not all(
            isinstance(state, GaussianState) for _, state in self.data
        ):  # Fock 0 should be included (both Gaussian and Fock)
            raise TypeError("All states in a LCGaussianState have to be GaussianState objects.")

        # add check normalisation and normalize
        # NOTE: this doesn't make sense since the states in the superposition are not orthogonal!
        # if not isclose(sqrt(sum([abs(coeff) ** 2 for coeff, _ in self.data])), 1):
        #     raise ValueError("The provided coefficients are not normalised.")

    def __iter__(self) -> Iterator[tuple[complex, GaussianState]]:
        """Iterate over the coefficient-state pairs.

        Yields
        ------
        tuple of (complex, GaussianState)
            Each coefficient and its associated Gaussian state.
        """
        return iter(self.data)

    # @functools.cache
    @typing_extensions.override  # toutes les filles qui implémentent get_statevector to check same signature
    def get_statevector(self, cutoff: int | None = None) -> Statevector:
        """returns the statevector of a `LCGaussianState` object by looping on its elements.

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

        .. math:: \vert
        """
        if not isinstance(cutoff, int):
            raise TypeError("The Fock space cutoff has to be an integer.")
        if not cutoff >= 0:
            raise ValueError("The Fock space cutoff has to be greater or equal than zero.")

        # needed convert to np.array
        tot_data = np.sum(
            np.array(
                [
                    coef * state.get_statevector(cutoff=cutoff).statevector.astype(np.complex128)
                    for coef, state in self.data
                ]
            ),
            axis=0,
        )

        return Statevector(tot_data)

    def __repr__(self) -> str:
        return f"LCGaussianState(data={self.data})"


# linter doesn't see init = False
@dataclass(frozen=True, init=False)
class CatState(LCGaussianState):  # type: ignore[misc]
    """A normalized superposition of two opposite coherent states.

    Parameters
    ----------
    amplitude : complex
        Coherent-state amplitude.
    parity : bool, default=False
        Selects the superposition sign: ``False`` gives even parity and
        ``True`` gives odd parity.

    Notes
    -----
    The two coherent components are normalized together, including their
    nonzero overlap.

    """

    amplitude: complex  # type: ignore
    parity: bool  # type: ignore
    # TODO: allow for an angle instead of just bool?

    def __init__(self, amplitude: complex, parity: bool = False) -> None:
        """Construct an even- or odd-parity cat state.

        Parameters
        ----------
        amplitude : complex
            Coherent-state amplitude.
        parity : bool, default=False
            ``False`` constructs the even superposition; ``True`` constructs
            the odd superposition.
        """
        # defaults to even parity
        norm = sqrt(2 * (1 + (-1) ** parity * exp(-2 * abs(amplitude) ** 2)))
        super().__init__(
            data=(
                (1.0 / norm, CoherentState(amplitude)),
                ((-1) ** parity / norm, CoherentState(-amplitude)),
            )
        )

        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "parity", parity)

    def __repr__(self) -> str:
        return f"CatState(amplitude={self.amplitude}, parity={self.parity})"


@dataclass(frozen=True, init=False)  # manually define __init__ for parameter order reasons
class BinomialState(PureCVState):
    """A single-mode binomial state in the Fock basis.

    Parameters
    ----------
    N : int
        Order controlling the number of populated Fock levels.
    S : int
        Spacing parameter between populated Fock levels.
    parity : bool, default=False
        Selects even (``False``) or odd (``True``) indices in the binomial
        superposition.
    """

    # todo name those parameters
    # max order
    N: int  # type: ignore
    # spacing
    S: int  # type: ignore
    parity: bool  # type: ignore

    def __init__(self, N: int, S: int, parity: bool = False):
        """Initialize a binomial state.

        Parameters
        ----------
        N : int
            Order parameter.
        S : int
            Spacing parameter.
        parity : bool, default=False
            Parity of the populated Fock levels.

        Raises
        ------
        TypeError
            If ``N`` or ``S`` is not an integer.
        """
        if not isinstance(N, int):
            raise TypeError("The maximal order has to be an integer.")
        if not isinstance(S, int):
            raise TypeError("The spacing has to be an integer.")
        else:
            super().__init__(statevector=None, is_gaussian=False)  # might raise pbs since might have only vac?
        object.__setattr__(self, "N", N)  # since frozen dataclass
        object.__setattr__(self, "S", S)  # since frozen dataclass
        object.__setattr__(self, "parity", parity)  # since frozen dataclass

    def __repr__(self) -> str:
        return f"BinomialState(N={self.N}, S={self.S}, parity={self.parity})"

    # @functools.cache
    @typing_extensions.override
    def get_statevector(self, cutoff: int | None = None) -> Statevector:
        """returns the statevector of a `BinomialState` object.

        Parameters
        ----------
        cutoff : int
            single-mode Fock space cutoff i.e. the highest Fock number reached
            # TODO put a default to the max. TODO default for Fockstates!

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
        """
        if not isinstance(cutoff, int):
            raise TypeError("The Fock space cutoff has to be an integer.")
        # computing the highest reached value: N + 1 or N?
        N_parity = (self.N + 1) % 2
        intrinsic_cutoff = self.N + 1

        if N_parity != self.parity:
            intrinsic_cutoff = self.N

        if not cutoff >= intrinsic_cutoff * (self.S + 1):
            raise ValueError(
                f"The Fock space cutoff has to be greater than the maximal Fock number reached by the state, here {intrinsic_cutoff * (self.S + 1)}."
            )

        indices = [(2 * k + self.parity) * (self.S + 1) for k in range((self.N + 1 - self.parity) // 2 + 1)]

        data = np.zeros(cutoff + 1, dtype=np.complex128)
        values = [sqrt(comb(self.N + 1, j // (self.S + 1))) for j in indices]

        np.put(data, indices, values)

        # TODO rewrite as full list comprehension?

        # TODO LCFockState useless since statevector?

        return Statevector(data / sqrt(2**self.N))


@dataclass(frozen=True, init=False)
class GKPState(LCGaussianState):  # type: ignore[misc]
    """An approximate square Gottesman-Kitaev-Preskil (GKP) state.

    The approximate comb replaces position eigenstates with finitely squeezed
    states and applies a Gaussian envelope.

    Parameters
    ----------
    dimension : int, default=2
        Logical-space dimension, at least two.
    index : int, default=0
        Logical-state index, conventionally in the range
        ``0 <= index < dimension``.
    kappa : float, default=0.3
        Gaussian envelope parameter.
    delta : float, default=0.3
        Width parameter for the squeezed states; must be strictly between zero
        and one.
    tol : float, default=1e-3
        Positive tolerance below one controlling the finite Gaussian expansion.

    Notes
    -----
    ``tol`` controls the state expansion itself and is distinct from the
    Fock-space cutoff passed to :meth:`get_statevector`.

    Raises
    ------
    TypeError
        If ``dimension`` is not an integer or a floating-point parameter is
        not a float.
    ValueError
        If ``dimension`` is less than two or ``delta`` or ``tol`` is outside
        its required open interval.

    """

    # TODO rename
    dimension: int  # type: ignore
    index: int  # type: ignore
    kappa: float  # type: ignore
    delta: float  # type: ignore
    tol: float  # type: ignore

    def __init__(
        self, dimension: int = 2, index: int = 0, kappa: float = 0.3, delta: float = 0.3, tol: float = 1e-3
    ) -> None:
        """Construct an approximate square GKP state.

        Parameters
        ----------
        dimension : int, default=2
            Logical-space dimension.
        index : int, default=0
            Logical-state index.
        kappa : float, default=0.3
            Gaussian envelope parameter.
        delta : float, default=0.3
            Width parameter for the squeezed states, strictly between zero and
            one.
        tol : float, default=1e-3
            Tolerance controlling the finite Gaussian expansion, strictly
            between zero and one.

        Raises
        ------
        TypeError
            If ``dimension`` is not an integer or one of the floating-point
            parameters is not a float.
        ValueError
            If ``dimension`` is below two, ``delta`` or ``tol`` is outside its
            required range.
        """
        """Construct an approximate square GKP state.

        Parameters
        ----------
        dimension : int, default=2
            Logical-space dimension.
        index : int, default=0
            Logical-state index.
        kappa : float, default=0.3
            Gaussian envelope parameter.
        delta : float, default=0.3
            Width parameter for the squeezed states, strictly between zero and
            one.
        tol : float, default=1e-3
            Tolerance controlling the finite Gaussian expansion, strictly
            between zero and one.

        Raises
        ------
        TypeError
            If ``dimension`` is not an integer or one of the floating-point
            parameters is not a float.
        ValueError
            If ``dimension`` is below two, ``delta`` or ``tol`` is outside its
            required range.
        """
        if not isinstance(dimension, int):
            raise TypeError(f"Logical space dimension has to be an integer, not {type(dimension)}.")
        if not dimension >= 2:
            raise ValueError(f"Logical space dimension has to greater or equal than two, not {dimension}.")
        if not isinstance(kappa, float):
            raise TypeError(f"Enveloppe parameter κ has to be a float, not {type(kappa)}.")
        if not isinstance(delta, float):
            raise TypeError(f"Squeezed state width Δ has to be a float, not {type(delta)}.")
        if not 0 < delta < 1:
            raise ValueError(f"Squeezed state width Δ has to be a greater than 0 and smaller than 1, not {delta}.")
        if not isinstance(tol, float):
            raise TypeError(f"Tolerance parameter must be a float, not {type(tol)}.")
        if not 0 < tol < 1:
            raise ValueError(f"Tolerance parameter must be between 0 and 1 (excluded), not {tol}.")

        # compute smax the cutoff in the Gaussian expansion
        # math.log is natural log
        # take ceil to be safe
        smax = ceil(sqrt(-log(tol) / (π * kappa**2 * dimension)))
        # without `tuple` this is a generator. More efficient? Don't think so the states carries it always.
        # log is natural log, no phase needed since 0 < delta < 1 => - ln delta > 0
        print(f"{smax=}")
        # TODO refactor for more efficiency

        coeffs = np.array(
            [exp(-π * kappa**2 * (dimension * s + index) ** 2 / dimension) for s in range(-smax, smax + 1)],
            dtype=np.float64,
        )
        norm = sqrt(np.sum(np.abs(coeffs) ** 2))

        data = tuple(
            (
                exp(-π * kappa**2 * (dimension * s + index) ** 2 / dimension) / norm,
                GaussianState(
                    GaussianParameters(x=sqrt(π / dimension) * (dimension * s + index), y=0, r=-log(delta), theta=0)
                ),
            )
            for s in range(-smax, smax + 1)
        )

        super().__init__(data=data)

        # NOTE could add attributes before...
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "tol", tol)

    def __repr__(self) -> str:
        return f"GKPState(dimension={self.dimension}, index={self.index}, kappa={self.kappa}, delta={self.delta}, tol={self.tol:.2e})"


@dataclass(frozen=True, init=False)
class TruncatedParityOp(HermitianCVOp):
    r"""Parity operator truncated in the Fock basis.

    Parameters
    ----------
    cutoff : int
        Highest Fock number included in the operator.

    Notes
    -----
    The operator is :math:`\Pi_n = \sum_{k=0}^n (-1)^k |k\rangle\langle k|`.
    """

    cutoff: int  # type: ignore

    def __init__(self, cutoff: int) -> None:
        """Construct parity truncated at the specified Fock number.

        Parameters
        ----------
        cutoff : int
            Highest Fock number included.

        Raises
        ------
        ValueError
            If ``cutoff`` is ``None``.
        """
        if cutoff is None:
            raise ValueError("cutoff cannot be None.")

        decomp: PureDecompositionData = tuple(((-1) ** k, FockState(n=k)) for k in range(cutoff + 1))
        super().__init__(data=decomp)
        object.__setattr__(self, "cutoff", cutoff)

    def __repr__(self) -> str:
        return f"TruncatedParityOp(cutoff={self.cutoff})"
