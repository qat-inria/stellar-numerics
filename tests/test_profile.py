from stellar.profile import compute_sup_fidelity
from stellar.states import StateFockBasis
import numpy as np
from math import e, isclose, sqrt


def test_optimize() -> None:
    # Fock state |2>
    tgt_state = StateFockBasis(np.array([0, 0, 1, 0]))
    results = compute_sup_fidelity(max_rank=3, target_state=tgt_state)
    print(f"{results.fun=}")
    assert results.success


# see [2] table IV and [3] Eq. (65)
def test_fock_state_1() -> None:
    # target |1> Fock state approximated using only Gaussian states
    tgt_state = StateFockBasis(np.array([0, 1, 0, 0]))
    results = compute_sup_fidelity(max_rank=0, target_state=tgt_state)
    # 1e-8 doesn't work
    assert isclose(results.fun, -3 * sqrt(3) / (4 * e), abs_tol=1e-7)

    # other tests: see table 4 of [2] for numerical values.
    # Other exact values might be derived for some photon number env 4

    # and test profiles
