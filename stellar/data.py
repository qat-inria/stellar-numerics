from typing import Generic, Iterator, TypeVar
from dataclasses import InitVar, dataclass, field
import matplotlib.pyplot as plt

# unused imports required for from_file method since can encounter all possible instances. use *?
from stellar.cvstates import (
    HermitianCVOp,
    PureCVState,
    CatState,  # noqa: F401
    GKPState,  # noqa: F401
    BinomialState,  # noqa: F401
    CoherentState,  # noqa: F401
    SqueezedVacuumState,  # noqa: F401
    GaussianState,  # noqa: F401
    LCGaussianState,  # noqa: F401
    FockState,  # noqa: F401
)
from stellar.params import OptimisationParameters, Method  # noqa: F401

from pathlib import Path
import json

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

    def __iter__(self) -> Iterator[tuple[int, float]]:  # TODO return type annotate this
        return iter(self.profile.items())

    def to_dict(self):  # TODO return type annotate this
        return {
            "state": repr(self.state),
            "profile": self.profile,
            "optim_params": repr(self.optim_params),
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
            json.dump(self.to_dict(), f, indent=4)

    # NOTE: how to avoid all possibilities in States?
    @staticmethod
    def from_file(
        filename: str, path: Path | None = None
    ):  # -> StellarProfile TODO how to type with generics without knowing?
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
        # # always
        path.mkdir(parents=True, exist_ok=True)

        with open(path / (filename + ".json"), "r") as f:
            d = json.load(f)

        # TODO `eval` is not safe! Do something better.
        return StellarProfile(
            eval(d["state"]),
            ranks=list(int(k) for k in d["profile"].keys()),
            fidelities=list(float(v) for v in d["profile"].values()),
            optim_params=eval(d["optim_params"]),
        )

    def draw(self, filename: str, path: Path | None = None, show: bool = False):
        """Method to generate a profile graph from the StellarProfile object"""

        if path is None:
            path = Path("tmp/profiles/")

        # Example data
        values = list(self.profile.values())

        # X positions (0, 1, 2, ...)
        x = list(self.profile.keys())

        plt.figure(figsize=(8, 5))

        # Plot vertical lines
        for i, v in zip(x, values):
            plt.vlines(x=i, ymin=0, ymax=v, color='r', linewidth=3)
            # Horizontal dashed line from y-axis to this bar
            plt.hlines(y=v, xmin=-1, xmax=i, linestyles='dashed', colors=['gray'], linewidth=1, alpha=0.7)
            plt.text(i, v + 0.04, str(v), ha='center', va='bottom', fontsize=10)

        # Configure axes
        plt.xticks(x)
        plt.xlim(-0.5, len(values) - 0.5)
        plt.ylim(0, 1.1)
        plt.xlabel('rank')
        plt.ylabel('Stellar fidelity')
        # plt.title('Stellar profile for ...')
        if show:
            plt.show()
        plt.savefig(path / (filename + ".pdf"))

