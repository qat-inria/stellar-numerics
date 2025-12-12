# builtin __init__ and __repr__
from dataclasses import dataclass
from cmath import exp
from enum import Enum, auto
import warnings


### Gaussian parameters
@dataclass(frozen=True)  # need frozen to implement __hash__ method for cacheing
class GaussianParameters:
    """Dataclass containing single-mode Gaussian parameters to be used with both `GaussianStates`and `GaussianOp`.
    NOTE r has to be positive? Or deal separately
        Returns
        -------
        _type_
            _description_
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
        return self.x + 1j * self.y

    @property
    def squeezing(self) -> complex:
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
    """A dataclass for recording and serializing optimization parameters"""

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
        if self.method == Method.fock:
            if self.target_cutoff is None:
                raise ValueError("cutoff cannot be None when computing in the Fock basis.")

        if self.method == Method.gaussian and self.target_cutoff is not None:
            warnings.warn("`target_cutoff` will be ignored using the `gaussian` method.")

    def to_dict(self):
        return {
            "state": repr(self.state),
            "ranks": self.ranks,
            "fidelities": self.fidelities,
            "optim_params": self.optim_params.to_dict(),
        }
