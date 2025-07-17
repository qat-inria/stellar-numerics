# builtin __init__ and __repr__
from dataclasses import dataclass
from cmath import exp


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
        if not self.r >= 0:
            raise ValueError("Parameter 'r' has to be non-negative.")
        if not isinstance(self.theta, (float, int)):
            raise TypeError("Parameter 'theta' has to be a float.")

    @property
    def displacement(self) -> complex:  # TODO: change to displacement
        return self.x + 1j * self.y

    @property
    def squeezing(self) -> complex:
        return self.r * exp(1j * self.theta)
