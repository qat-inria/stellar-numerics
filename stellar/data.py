from typing import Any, Generic, Iterator, TypeVar
from dataclasses import InitVar, dataclass, asdict, field, is_dataclass

from stellar.cvstates import HermitianCVOp, PureCVState
from stellar.params import OptimisationParameters

from pathlib import Path
import json


def encode(obj: Any) -> Any:
    if isinstance(obj, (PureCVState, HermitianCVOp)):
        print("in encode")
        return repr(obj)
    if is_dataclass(obj):
        print("in encode2")
        return asdict(obj)


# obj = MyData(1, Nested(42, some_state))
# jsonable = asdict(obj, dict_factory=lambda d: {k: encode(v) if isinstance(v, State) else v for k, v in d.items()})
# Result: {'x': 1, 'nested': {'value': 42, 'state': 'State(repr_here)'}}
# json.dump(jsonable, fp)

# {k: encode(v) if isinstance(v, State) else v for k, v in fields_list}

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
    # state: InitVar[S]  # constructor param name = c
    state: S  # PureCVState | HermitianCVOp
    # _state: str = field(init=False)  # stored attribute
    ranks: InitVar[list[int]]
    fidelities: InitVar[list[float]]
    profile: dict[int, float] = field(init=False)
    optim_params: OptimisationParameters | None = None  # TODO remove None

    def __post_init__(self, ranks: list[int], fidelities: list[float]) -> None:  # , state: S
        # object.__setattr__(self, "_state", repr(state))  # same trick since frozen dataclass
        if len(ranks) != len(fidelities):
            raise ValueError("The length of ranks and fidelities have to match.")
        object.__setattr__(self, "profile", dict(zip(ranks, fidelities)))  # same trick since frozen dataclass
        # self.profile = dict(zip(ranks, fidelities))

    def __iter__(self) -> Iterator:  # TODO return type annotate this
        return iter(self.profile.items())

    def to_dict(self):  # TODO return type annotate this
        return {
            "state": repr(self.state),
            "ranks": self.ranks,
            "fidelities": self.fidelities,
            "optim_params": asdict(self.optim_params),
        }

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
            json.dump(self.to_dict(), f)


# asdict(self, dict_factory=lambda fields: {k: encode(v) if isinstance(v, (PureCVState, HermitianCVOp)) else v for k, v in fields}), f)
