from math import isclose
import os
from pathlib import Path

import numpy as np
import pytest

from stellar.cvstates import BinomialState, CatState, CoherentState, FockState, PureCVState
from stellar.data import StellarProfile
from stellar.params import Method, OptimisationParameters
from stellar.profile import compute_profile


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
    prof = StellarProfile(
        state=st,
        ranks=list(range(3)),
        fidelities=[0.1, 0.2, 0.5],
        optim_params=OptimisationParameters(method=Method.gaussian),
    )
    # , optim_params=OptimisationParameters(method=Method.gaussian)
    # TODO serialize OptimParams too
    prof.save_to_file(filename="dummyb")


def test_deserialization() -> None:
    st = CatState(amplitude=3)
    pars = OptimisationParameters(method=Method.gaussian)
    profile = compute_profile(ranks=list(range(1)), target_state=st, optim_params=pars)

    path = Path("tests/data/")
    path.mkdir(parents=True, exist_ok=True)

    profile.save_to_file("test", path=path)

    prof = StellarProfile.from_file("test", path)

    assert isinstance(prof.state, CatState)
    assert prof.state.amplitude == st.amplitude
    assert prof.state == st
    assert prof.optim_params == pars
    assert isclose(prof.profile[0], 0.5, abs_tol=1e-8)


def test_draw() -> None:
    "dummy test"
    st = BinomialState(N=3, S=2)
    rng = np.random.default_rng(seed=42)
    prof = StellarProfile(
        state=st,
        ranks=list(range(15)),
        fidelities=sorted(list(rng.random(15))),
        optim_params=OptimisationParameters(method=Method.gaussian),
    )

    # , optim_params=OptimisationParameters(method=Method.gaussian)
    # TODO serialize OptimParams too

    prof.draw(filename="dummyb", text=False)

    assert os.path.exists(Path("tmp/profiles/") / ("dummyb" + ".pdf"))
