from stellar.cvstates import BinomialState, CoherentState, FockState, PureCVState
from stellar.data import StellarProfile
import pytest

from stellar.params import OptimisationParameters, Method


def test_init_success() -> None:
    st = FockState(n=3)

    prof = StellarProfile(st, list(range(3)), [0.1, 0.2, 0.5])
    assert isinstance(prof.state, PureCVState)


def test_init_fail() -> None:
    st = CoherentState(amplitude=-0.8 - 0.3j)
    with pytest.raises(ValueError):
        StellarProfile(st, list(range(3)), [0.1, 0.2])


def test_iter() -> None:
    rs = list(range(3))
    fs = [0.1, 0.2, 0.5]
    prof = StellarProfile(state=FockState(n=3), ranks=rs, fidelities=fs)

    for i, (r, f) in enumerate(prof):
        # print(r, f)
        assert r == rs[i]
        assert f == fs[i]


def test_serialization() -> None:
    st = BinomialState(N=3, S=2)
    prof = StellarProfile(state=st, ranks=list(range(3)), fidelities=[0.1, 0.2, 0.5])
    # , optim_params=OptimisationParameters(method=Method.gaussian)
    # TODO serialize OptimParams too
    prof.save_to_file(filename="dummyb")
