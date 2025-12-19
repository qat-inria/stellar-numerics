"""test conversion module"""

from math import isclose, sqrt
import time
from stellar.conversion import max_trace_distance_precision_pure_pure_post, max_trace_distance_precision_pure_pure_std
from stellar.cvstates import CatState, CoherentState, GKPState
from stellar.params import Method, OptimisationParameters
from stellar.profile import compute_profile
import pytest


def test_det_conversion() -> None:
    from_state = CoherentState(amplitude=3)
    to_state = CatState(amplitude=1, parity=True)

    max_rank = 10
    # same parameters for both states
    pars = OptimisationParameters(method=Method.gaussian)

    from_profile = compute_profile(ranks=list(range(max_rank)), target_state=from_state, optim_params=pars)
    to_profile = compute_profile(ranks=list(range(max_rank)), target_state=to_state, optim_params=pars)
    max_dist = max_trace_distance_precision_pure_pure_std(from_profile=from_profile, to_profile=to_profile, nb_copies=2)
    print(f"{max_dist=}")  # need 1 - f_0(cat)
    # assert False


@pytest.mark.skip
def test_det_conversion_paper() -> None:
    from_state = CatState(amplitude=1, parity=True)  # 6
    to_state = GKPState(delta=0.3, kappa=0.3)  # more terms (11 vs 4) than Δ = κ = 0.3

    max_rank = 1  # why 20 profile file?
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


def test_det_conversion_prob() -> None:
    """test conversion from rank 1 to even cat state amp √4"""
    # check for 3 copies with Rui's value: 1-f_3(cat) = 1-0.896384 ≅ 0.103616
    # from_state = CoherentState(amplitude=3)
    true_value = 1 - 0.896384
    to_state = CatState(
        amplitude=3, parity=False
    )  # sqrt(4) gives 0.021810 kind of ok conversion paper [HFFC25] Fig. 4.

    max_rank = 5
    # same parameters for both states
    pars = OptimisationParameters(method=Method.gaussian)

    message = "For postselected gaussian pure state conversion, the `from_profile` is not used and only depends on the input state's stellar rank."
    to_profile = compute_profile(ranks=list(range(max_rank)), target_state=to_state, optim_params=pars)
    with pytest.warns(match=message):
        max_dist = max_trace_distance_precision_pure_pure_post(to_profile=to_profile, from_rank=1, nb_copies=3)
    print(f"{max_dist=}")  # need 1 - f_0(cat)
    assert isclose(max_dist, true_value, abs_tol=1e-5)
    # assert False
