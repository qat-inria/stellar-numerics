from stellar.cvstates import BinomialState, CoherentState, FockState
from stellar.data import StellarProfile
import pytest


def test_init_success() -> None:
    st = FockState(n=3)

    StellarProfile(repr(st), list(range(3)), [0.1, 0.2, 0.5])


def test_init_fail() -> None:
    st = CoherentState(amplitude=-0.8 - 0.3j)
    with pytest.raises(ValueError):
        StellarProfile(repr(st), list(range(3)), [0.1, 0.2])


def test_serialization() -> None:
    st = BinomialState(N=3, S=2)
    prof = StellarProfile(repr(st), list(range(3)), [0.1, 0.2, 0.5])
    prof.save_to_file(filename="dummy")

