# builtin __init__ and __repr__
import warnings
from cmath import exp
from dataclasses import dataclass
from enum import Enum, auto


### Gaussian parameters
@dataclass(frozen=True)  # need frozen to implement __hash__ method for cacheing
class GaussianParameters:
    """Parameters describing a single-mode Gaussian state or operation.

    Parameters
    ----------
    x, y : float
        Real and imaginary parts of the displacement, respectively.
    r : float
        Non-negative squeezing magnitude.
    theta : float
        Squeezing phase in radians.

    Notes
    -----
    Instances are immutable and hashable.
    """

    x: float  # real part of displacement
    y: float  # imaginary part of displacement
    r: float  # modulus of squeezing
    theta: float  # phase of sueezing

    def __post_init__(self) -> None:
        if not isinstance(self.x, (float, int)):
            raise TypeError("Parameter 'x' has to be a float.")
        if not isinstance(self.y, (float, int)):
            raise TypeError("Parameter 'y' has to be a float.")
        if not isinstance(self.r, (float, int)):
            raise TypeError("Parameter 'r' has to be a float.")
        ## NOTE TODO doesn't work so far since need to add constraints in the optimisation algorithm
        if not self.r >= 0:
            raise ValueError("Parameter 'r' has to be non-negative.")
        if not isinstance(self.theta, (float, int)):
            raise TypeError("Parameter 'theta' has to be a float.")

    @property
    def displacement(self) -> complex:
        """complex: Displacement amplitude ``x + 1j*y``."""
        return self.x + 1j * self.y

    @property
    def squeezing(self) -> complex:
        """complex: Complex squeezing parameter ``r * exp(1j*theta)``."""
        return self.r * exp(1j * self.theta)


### Optimisation parameters
#
#
# enum for optimisation methods: ie the way of computing
# change name OptimMethod?
class Method(Enum):
    """Enumeration of the methods to compute the objective function. Only 2 so far."""

    fock = auto()
    gaussian = auto()


@dataclass(frozen=True)
class OptimisationParameters:
    """Configuration for computing stellar fidelities and profiles.

    Parameters
    ----------
    method : Method
        Representation used for the optimization objective: Fock-basis or
        Gaussian-state evaluation.
    target_cutoff : int or None, optional
        Highest Fock number used to represent the target in the Fock basis.
        Required when ``method`` is :attr:`Method.fock`; ignored for
        :attr:`Method.gaussian`.
    niter : int, default=250
        Number of basin-hopping iterations.
    x0 : tuple of float, default=(0.1, 0.1, 0.1, 0.1)
        Initial values for displacement components, squeezing magnitude, and
        squeezing phase.
    seed : int or None, optional
        Random seed passed to the optimizer.

    Raises
    ------
    ValueError
        If Fock-basis optimization is selected without ``target_cutoff``.

    Warns
    -----
    UserWarning
        If ``target_cutoff`` is supplied for Gaussian-state optimization.
    """

    # want: method (gaussian or Fock), niter, starting point, rng (seed) other kwargs?
    # feed that to the compute_profile fct (to write)

    # TODO update for mixed states? or carried by the state?
    method: Method  # TODO use Enums as before
    target_cutoff: int | None = None
    niter: int = 250
    x0: tuple[float, ...] = (0.1,) * 4
    seed: int | None = None
    # other_kwargs: dict

    def __post_init__(self) -> None:
        if self.method == Method.fock and self.target_cutoff is None:
            raise ValueError("cutoff cannot be None when computing in the Fock basis.")

        if self.method == Method.gaussian and self.target_cutoff is not None:
            warnings.warn("`target_cutoff` will be ignored using the `gaussian` method.")

    def __repr__(self) -> str:
        return f"OptimisationParameters(method={self.method}, target_cutoff={self.target_cutoff}, niter={self.niter}, x0={self.x0}, seed={self.seed})"
