from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

import matplotlib.pyplot as plt

# unused imports required for from_file method since can encounter all possible instances. use *?
from stellar.cvstates import (
    BinomialState,  # noqa: F401
    CatState,  # noqa: F401
    CoherentState,  # noqa: F401
    FockState,  # noqa: F401
    GaussianState,  # noqa: F401
    GKPState,  # noqa: F401
    HermitianCVOp,
    LCGaussianState,  # noqa: F401
    PureCVState,
    SqueezedVacuumState,  # noqa: F401
)
from stellar.params import Method, OptimisationParameters  # noqa: F401

# obj = MyData(1, Nested(42, some_state))
# jsonable = asdict(obj, dict_factory=lambda d: {k: encode(v) if isinstance(v, State) else v for k, v in d.items()})
# Result: {'x': 1, 'nested': {'value': 42, 'state': 'State(repr_here)'}}
# json.dump(jsonable, fp)

# {k: encode(v) if isinstance(v, State) else v for k, v in fields_list}

# TODO use a mapping instead of two separate lists
# quick fix: dict(zip(ranks, fidelities))
# TODO add __iter__ function
# Todo add draw( function)
S_co = TypeVar("S_co", bound=PureCVState | HermitianCVOp, covariant=True)
S = TypeVar("S", bound=PureCVState | HermitianCVOp)
# covariance on variables for classes that are parametrised by them
# comparison on S allows to deduce comparaison on  StellarProfile[S]]

# can return covariant type


@dataclass(frozen=True, init=False)
class StellarProfile(Generic[S_co]):
    """A stellar-fidelity profile with its state and optimization metadata.

    Parameters
    ----------
    state : PureCVState or HermitianCVOp
        State for which the profile was computed.
    ranks : iterable of int
        Stellar ranks corresponding to the fidelity values.
    fidelities : iterable of float
        Fidelity at each corresponding rank.
    optim_params : OptimisationParameters or None, optional
        Parameters used to compute the profile, if available.

    Raises
    ------
    ValueError
        If ``ranks`` and ``fidelities`` have different lengths.
    """

    # single rank returns a StellarProfile?
    # Combine them by concatenation if different ranks but same state and params.
    # state name of the parameter and _state for the field
    # state: InitVar[S]  # constructor param name = c
    state: S_co = field(init=False)  # PureCVState | HermitianCVOp
    # _state: str = field(init=False)  # stored attribute
    ranks: tuple[int]
    fidelities: tuple[float]
    profile: Mapping[int, float]
    optim_params: OptimisationParameters | None  # TODO remove None

    def __init__(
        self,
        state: S_co,
        ranks: Iterable[int],
        fidelities: Iterable[float],
        optim_params: OptimisationParameters | None = None,
    ) -> None:
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "ranks", tuple(ranks))
        object.__setattr__(self, "fidelities", tuple(fidelities))
        object.__setattr__(self, "optim_params", optim_params)
        if len(self.ranks) != len(self.fidelities):
            raise ValueError("The length of ranks and fidelities have to match.")
        profile = dict(zip(ranks, fidelities))
        object.__setattr__(self, "profile", profile)

    def __iter__(self) -> Iterator[tuple[int, float]]:  # TODO return type annotate this
        """Iterate over ``(rank, fidelity)`` pairs in profile order.

        Yields
        ------
        tuple of (int, float)
            Stellar rank and its fidelity.
        """
        return iter(self.profile.items())

    def to_dict(self):  # TODO return type annotate this
        """Return a JSON-serializable representation of the profile.

        Returns
        -------
        dict
            Mapping containing the state representation, rank-to-fidelity
            mapping, and optimization-parameter representation.
        """
        return {
            "state": repr(self.state),
            "profile": self.profile,
            "optim_params": repr(self.optim_params),
        }

    def replace(self, state: S) -> StellarProfile[S]:
        """Return a profile with a different associated state.

        Parameters
        ----------
        state : PureCVState or HermitianCVOp
            State to associate with the existing ranks and fidelities.

        Returns
        -------
        StellarProfile
            New profile retaining this profile's ranks, fidelities, and
            optimization parameters.
        """
        return StellarProfile(state, self.ranks, self.fidelities, self.optim_params)

    def save_to_file(self, filename: str, path: Path | None = None) -> None:
        """Serialize this profile as an indented JSON file.

        Parameters
        ----------
        filename : str
            File name without the ``.json`` extension.
        path : pathlib.Path or None, optional
            Directory in which to save the file. Parent directories are created
            as needed. Defaults to ``tmp/profiles/``.
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
        """Load a profile from a JSON file.

        Parameters
        ----------
        filename : str
            File name without the ``.json`` extension.
        path : pathlib.Path or None, optional
            Directory containing the file. Defaults to ``tmp/profiles/``.

        Returns
        -------
        StellarProfile
            Deserialized profile.

        Notes
        -----
        The current deserializer evaluates representations stored in the JSON.
        Only load profile files from trusted sources.
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
            ranks=[int(k) for k in d["profile"]],
            fidelities=[float(v) for v in d["profile"].values()],
            optim_params=eval(d["optim_params"]),
        )

    def draw(self, filename: str, path: Path | None = None, text: bool = True, show: bool = False):
        """Plot the profile and save it as a PDF.

        Parameters
        ----------
        filename : str
            Output file name without the ``.pdf`` extension.
        path : pathlib.Path or None, optional
            Output directory. Defaults to ``tmp/profiles/``.
        text : bool, default=True
            Whether to annotate each fidelity value above its plotted line.
        show : bool, default=False
            Whether to display the plot after saving it.

        Notes
        -----
        The method writes ``<path>/<filename>.pdf``. The output directory must
        already exist.
        """

        if path is None:
            path = Path("tmp/profiles/")

        # Example data
        values = list(self.profile.values())

        # X positions (0, 1, 2, ...)
        x = list(self.profile.keys())

        plt.figure(figsize=(8, 5))

        # Plot vertical lines
        for i, v in zip(x, values):
            plt.vlines(x=i, ymin=0, ymax=v, color="r", linewidth=3)
            # Horizontal dashed line from y-axis to this bar
            plt.hlines(y=v, xmin=-1, xmax=i, linestyles="dashed", colors=["gray"], linewidth=1, alpha=0.7)
            if text:
                plt.text(i, v + 0.04, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

        # Configure axes
        plt.xticks(x)
        plt.xlim(-0.5, len(values) - 0.5)
        plt.ylim(0, 1.1)
        plt.xlabel("rank")
        plt.ylabel("stellar fidelity")
        # plt.title('Stellar profile for ...')

        plt.savefig(path / (filename + ".pdf"))
        if show:
            plt.show()
