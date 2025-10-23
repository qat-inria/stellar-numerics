from stellar.cvstates import BinomialState, CatState, CoherentState, FockState, GKPState
from stellar.profile import compute_sup_fidelity
import numpy as np
from math import e, isclose, sqrt


def test_optimize() -> None:
    # Fock state |2>
    # tgt_state = StateFockBasis(np.array([0, 0, 1, 0]))
    tgt_state = FockState(n=2)
    results = compute_sup_fidelity(max_rank=3, target_state=tgt_state, target_cutoff=4)
    print(f"{results.fun=}")
    assert results.success


# see [2] table IV and [3] Eq. (65)
def test_fock_state_1() -> None:
    # target |1> Fock state approximated using only Gaussian states
    tgt_state = FockState(n=1)
    results = compute_sup_fidelity(max_rank=0, target_state=tgt_state, target_cutoff=4)
    # 1e-8 doesn't work
    assert isclose(results.fun, -3 * sqrt(3) / (4 * e), abs_tol=1e-5)

    results = compute_sup_fidelity(max_rank=1, target_state=tgt_state, target_cutoff=4)
    print(f"{results.fun=} {results.x} {results.nfev}")
    assert isclose(results.fun, -1)


# rank = 0 works but not higher values!
# 1 for |1> but not 2 for |2> indep of cutoff == ok
# dependenci on local optmizer starting point!
# ok made work for 2 for |2> with all 0 starting point of for 3 and |3> too.
# pick best of number of random starting points?
# made 1 for |2> work ith all zero starting point


def test_fock_state_2() -> None:
    # target |1> Fock state approximated using only Gaussian states
    tgt_state = FockState(n=2)

    # works for (0, 1, 3, .43) starting point
    results = compute_sup_fidelity(max_rank=0, target_state=tgt_state, target_cutoff=4)
    print(f"{results.fun=} {results.x} {results.success}")
    assert isclose(results.fun, -0.381, abs_tol=1e-3)

    # works for all (0,) * 4 starting point
    results = compute_sup_fidelity(max_rank=1, target_state=tgt_state, target_cutoff=4)
    print(f"{results.fun=} {results.x} {results.success}")
    assert isclose(results.fun, -0.557, abs_tol=1e-3)

    # works will al zero starting point
    results = compute_sup_fidelity(max_rank=2, target_state=tgt_state, target_cutoff=4)
    print(f"{results.fun=} {results.x} {results.nfev} {results.success}")
    assert isclose(results.fun, -1, abs_tol=1e-7)


def test_fock_states() -> None:
    # numerical results from see [2] table IV
    # don't use analytical for precision results

    # matrix first index is fock number second is rank
    check_results = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0.478, 1, 1, 1, 1, 1],
            [0.381, 0.557, 1, 1, 1, 1],
            [0.333, 0.462, 0.593, 1, 1, 1],
            [0.301, 0.409, 0.501, 0.612, 1, 1],
            [0.279, 0.374, 0.449, 0.525, 0.626, 1],
        ]
    )
    assert check_results.shape == (6, 6)

    cutoff = 7  # min is 6 for Fock state |5>

    for n in range(0, 6):
        tgt_state = FockState(n=n)
        for r in range(0, 6):
            results = compute_sup_fidelity(max_rank=r, target_state=tgt_state, target_cutoff=cutoff)
            # assert results.success?
            assert isclose(results.fun, -check_results[n, r], abs_tol=1e-3)


# assert isclose(results.fun, -3 * sqrt(3) / (4 * e), abs_tol=1e-7)
# other tests: see table 4 of [2] for numerical values.
# Other exact values might be derived for some photon number env 4


# reduce size a bit
# hypothesis gives some trouble due to test deadline
# TODO fix this
# BUG for alpha = 8 the squeezing in gauss matrix product overflows the atanh fct
# @given(st.complex_numbers(allow_nan=False, allow_infinity=False, min_magnitude=0, max_magnitude=5))
# @given(st.complex_numbers(allow_nan=False, allow_infinity=False, min_magnitude=0, max_magnitude=5))
def test_coh_state() -> None:  # amp: complex
    # target |1> Fock state approximated using only Gaussian states

    amp = 1  # 6.5-.529j
    tgt_state = CoherentState(amplitude=amp)

    results = compute_sup_fidelity(max_rank=0, target_state=tgt_state, target_cutoff=10)  # , method='g'
    print(f"{amp=} {results.fun=} {results.x} {results.success}")
    assert isclose(results.fun, -1, abs_tol=1e-6)


def test_run_cat_state() -> None:  # amp: complex
    amp = 6.5 - 0.529j
    tgt_state = CatState(amplitude=amp, parity=False)

    results = compute_sup_fidelity(max_rank=4, target_state=tgt_state, method="g")  # target_cutoff=25
    print(f"{amp=} {results.fun=} {results.x} {results.success}")
    # assert isclose(results.fun, -1, abs_tol=1e-6)
    # assert False

    # TODO here for new tests


def test_values_cat_state() -> None:  # amp: complex
    """Check againt Rui's (Chalmers) values: [0.5, 0.615383 ,0.848799, 0.896384]. Only discrepency is for rank 0"""
    # works with starting point (0,) * 4

    amp = 3
    tgt_state = CatState(amplitude=amp, parity=False)

    values = [0.5, 0.615383, 0.848799, 0.896384]

    for rank in range(0, 4):
        results = compute_sup_fidelity(max_rank=rank, target_state=tgt_state, method="g")
        assert isclose(results.fun, -values[rank], abs_tol=1e-6)


# and test profiles
# and add seed in opt


def test_values_bin_e21_state() -> None:  # amp: complex
    """check binomial stell W(even, N = 2, S = 1) state in [CDraft]"""
    # works with starting point (0,) * 4

    tgt_state = BinomialState(N=2, S=1)

    for rank in range(0, 5):
        results = compute_sup_fidelity(max_rank=rank, target_state=tgt_state, target_cutoff=6)
        print(f"results for {rank=}: {-results.fun}")


def test_values_gkp_state_default() -> None:  # amp: complex
    """check standard GKP state (d = 2, index = 0, delta = kappa = 0.3, tol = 1e-3) in [CDraft]"""

    # numerical results to test non regression
    # niter = 350 tol = 1e-3 (smax=4)
    # results for rank=0: 0.5998054164771314
    # results for rank=1: 0.5998054162670581
    # results for rank=2: 0.599805416470219
    # results for rank=3: 0.6224691842120489
    # results for rank=4: 0.7236673372401499
    # results for rank=5: 0.7290038000488364

    targets = [
        0.5998054164771314,
        0.5998054162670581,
        0.599805416470219,
        0.6224691842120489,
        0.7236673372401499,
        0.7290038000488364,
    ]
    tgt_state = GKPState(tol=1e-3)

    for rank in range(0, 6):
        # seed for test reproducibility
        results = compute_sup_fidelity(max_rank=rank, target_state=tgt_state, method="g", rng=42, niter=350)
        print(f"results for {rank=}: {-results.fun}")

        assert isclose(-results.fun, targets[rank], abs_tol=1e-8)

    # assert False
