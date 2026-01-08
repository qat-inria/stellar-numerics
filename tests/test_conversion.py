"""test conversion module"""

from math import isclose
import time
from stellar.conversion import (
    max_trace_distance_precision,
    max_trace_distance_precision_pure_pure_post,
    max_trace_distance_precision_pure_pure_std,
    Protocol,
)
from stellar.cvstates import CatState, CoherentState, FockState, GKPState, HermitianCVOp, PureDecompositionData
from stellar.params import Method, OptimisationParameters
from stellar.profile import compute_profile
import pytest

# TODO add pytest fixtures for commonly used profiles


def test_det_conversion_fail() -> None:
    # from_state = CoherentState(amplitude=3)
    to_state = CatState(amplitude=1, parity=True)

    max_rank = 1
    # same parameters for both states
    pars = OptimisationParameters(method=Method.gaussian)

    # from_profile = compute_profile(ranks=list(range(max_rank)), target_state=from_state, optim_params=pars)
    # TODO could use a dummy profile here to accelerate testing phase
    to_profile = compute_profile(ranks=list(range(max_rank)), target_state=to_state, optim_params=pars)

    with pytest.raises(ValueError, match="Both `from_profile`and `from_rank` cannot be None."):
        max_trace_distance_precision(protocol=Protocol.standard, nb_copies=2, to_profile=to_profile)

    with pytest.raises(
        ValueError,
        match="`from_profile` cannot be None when assessing Gaussian conversion with a non-postselected protocol.",
    ):
        max_trace_distance_precision(protocol=Protocol.standard, nb_copies=2, to_profile=to_profile, from_rank=3)


def test_det_conversion_fail_mixed() -> None:
    """check that targetting a mixed state raises an error"""

    decomp: PureDecompositionData = ((0.5, FockState(n=1)), (0.5, FockState(n=2)))  # don't forget the comma!
    to_state = HermitianCVOp(data=decomp)
    from_state = CoherentState(amplitude=3)

    max_rank = 10
    # same parameters for both states
    pars = OptimisationParameters(method=Method.fock, target_cutoff=4)

    from_profile = compute_profile(ranks=list(range(max_rank)), target_state=from_state, optim_params=pars)

    to_profile = compute_profile(ranks=list(range(max_rank)), target_state=to_state, optim_params=pars)

    with pytest.raises(ValueError, match="Cannot assess Gaussian conversion to a mixed state."):
        # manually disable typing warning since want to check the dynamical error
        max_trace_distance_precision(
            protocol=Protocol.standard,
            nb_copies=2,
            to_profile=to_profile,
            from_profile=from_profile,  # type: ignore
        )


def test_det_conversion() -> None:
    from_state = CoherentState(amplitude=3)
    to_state = CatState(amplitude=1, parity=False)

    max_rank = 10
    # same parameters for both states
    pars = OptimisationParameters(method=Method.gaussian)

    from_profile = compute_profile(ranks=list(range(max_rank)), target_state=from_state, optim_params=pars)
    to_profile = compute_profile(ranks=list(range(max_rank)), target_state=to_state, optim_params=pars)
    max_dist = max_trace_distance_precision_pure_pure_std(from_profile=from_profile, to_profile=to_profile, nb_copies=2)
    print(f"{max_dist=} and {1 - to_profile.profile[0]}")  # need 1 - f_0(cat)

    assert isclose(max_dist, 1 - to_profile.profile[0], abs_tol=1e-8)

    max_dist_2 = max_trace_distance_precision(
        protocol=Protocol.standard, nb_copies=2, to_profile=to_profile, from_profile=from_profile
    )
    print(f"{max_dist=} and {max_dist_2}")
    assert isclose(max_dist, max_dist_2)
    # assert False


@pytest.mark.skip
def test_det_conversion_paper() -> None:
    from_state = CatState(amplitude=1, parity=False)  # 6
    to_state = GKPState(delta=0.3, kappa=0.3)  # more terms (11 vs 4) than Δ = κ = 0.3

    max_rank = 1
    # same parameters for both states
    pars = OptimisationParameters(method=Method.gaussian, niter=350)
    init = time.time()
    from_profile = compute_profile(ranks=list(range(max_rank)), target_state=from_state, optim_params=pars)
    from_time = time.time()
    print(f"from profile time {from_time - init}")
    to_profile = compute_profile(ranks=list(range(max_rank)), target_state=to_state, optim_params=pars)
    print(f"to_profile {to_profile.profile}")
    to_time = time.time()
    print(f"to profile time {to_time - from_time}")

    from_profile.save_to_file("test")
    to_profile.save_to_file("gkp33350")
    # with open("gkp33350" + ".json", "w") as f:
    #     json.dump(to_profile.profile, f)
    dist_time = time.time()
    max_dist = max_trace_distance_precision_pure_pure_std(from_profile=from_profile, to_profile=to_profile, nb_copies=1)
    print(f"{max_dist=}")  # expect something around 0.23
    print(f"dist time {dist_time - to_time}")
    assert False


def test_det_conversion_post() -> None:
    """test conversion from rank 1 to even cat state amp"""
    # check for 3 copies with Rui's value: 1-f_3(cat) = 1-0.896384 ≅ 0.103616
    # from_state = CoherentState(amplitude=3)
    true_value = 1 - 0.896384
    to_state = CatState(
        amplitude=3, parity=False
    )  # sqrt(4) = √4 gives 0.021810 kind of ok conversion paper [HFFC25] Fig. 4.

    max_rank = 5
    # same parameters for both states
    pars = OptimisationParameters(method=Method.gaussian)

    to_profile = compute_profile(ranks=list(range(max_rank)), target_state=to_state, optim_params=pars)

    max_dist = max_trace_distance_precision_pure_pure_post(to_profile=to_profile, from_rank=1, nb_copies=3)
    message = "`from_profile` is not used in assessing Gaussian conversion with a postselected protocol."

    with pytest.raises(
        ValueError,
        match="`from_rank` has to be specified in assessing Gaussian conversion with a postselected protocol",
    ):
        # use from_profile = to_profile to check the warning
        max_trace_distance_precision(Protocol.postselected, nb_copies=3, to_profile=to_profile, from_profile=to_profile)

    with pytest.warns(match=message):
        # use from_profile = to_profile to check the warning
        max_dist_2 = max_trace_distance_precision(
            Protocol.postselected, nb_copies=3, to_profile=to_profile, from_profile=to_profile, from_rank=1
        )
    print(f"{max_dist=}")  # need 1 - f_3(cat)
    print(f"{true_value=}")
    print(f"{max_dist_2=}")
    assert isclose(max_dist, true_value, abs_tol=1e-5)
    assert isclose(max_dist, max_dist_2, abs_tol=1e-8)
    # assert False
