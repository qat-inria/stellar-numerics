from typing import Generic, TypeVar
from dataclasses import InitVar, dataclass, asdict, field

from stellar.cvstates import HermitianCVOp, PureCVState
from stellar.params import OptimisationParameters

from pathlib import Path
import json


# TODO use a mapping instead of two separate lists
# quick fix: dict(zip(ranks, fidelities))
# TODO add __iter__ function
# Todo add draw( function)
S = TypeVar("S", bound=PureCVState | HermitianCVOp)


@dataclass(frozen=True)
class StellarProfile(Generic[S]):
    """A dataclass for stellar profiles allowing manipulation and serialization to json."""

    # single rank returns a StellarProfile?
    # Combine them by concatenation if different ranks but same state and params.
    # state name of the parameter and _state for the field
    state: InitVar[S]  # constructor param name = c
    _state: str = field(init=False)  # stored attribute
    ranks: list[int]
    fidelities: list[float]
    optim_params: OptimisationParameters | None = None  # TODO remove None

    def __post_init__(self, state: S) -> None:
        object.__setattr__(self, "_state", repr(state))  # same trick since frozen dataclass
        if len(self.ranks) != len(self.fidelities):
            raise ValueError("The length of ranks and fidelities have to match.")

    def save_to_file(self, filename: str, path: Path | None = None) -> None:
        """
        save to file a given stellar profile by specifying a filename and a path.

        Parameters
        ----------
        filename : str
            title of the file

        path : pathlib.Path | None, optional
            path to the folder to save in. If not created, the whole hierarchy will be created.
            default: None i.e. tmp/profiles/
        """

        if path is None:
            path = Path("tmp/profiles/")
        # always
        path.mkdir(parents=True, exist_ok=True)

        with open(path / (filename + ".json"), "w") as f:
            json.dump(asdict(self), f)
