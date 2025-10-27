from dataclasses import dataclass, asdict

from stellar.params import OptimisationParameters

from pathlib import Path
import json


@dataclass(frozen=True)
class StellarProfile:
    """A dataclass for stellar profiles allowing manipulation and serialization to json."""

    # attributes: state, list of ranks, stellar fidelities, optim params
    # add compute profile
    # single rank returns a StellarProfile?
    # Combine them by concatenation if different ranks but same state and params.
    state: str  # repr(State) or directly state and take repr? no init then
    ranks: list[int]
    fidelities: list[float]
    optim_params: OptimisationParameters | None = None # TODO remove None

    def __post_init__(self) -> None:
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

        with open(path / (filename + '.json'), 'w') as f:
            json.dump(asdict(self), f)

