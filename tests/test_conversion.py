"""test conversion module"""

from math import isclose
from stellar.conversion import (
    max_trace_distance_precision,
    max_trace_distance_precision_mixed_pure_std,
    max_trace_distance_precision_pure_pure_post,
    max_trace_distance_precision_pure_pure_std,
    Protocol,
)
from stellar.cvstates import CatState, CoherentState, FockState, HermitianCVOp, PureDecompositionData
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
        # manually disable typing error since want to check the dynamical error
        # Indeed, `to_profile` variable has type `StellarProfile[HermitianCVOp]` whereas
        #   `to_profile` parameter expects the type `StellarProfile[PureCVState]`,
        #    and `HermitianCVOp` is not a subtype of `PureCVState`.
        max_trace_distance_precision(
            protocol=Protocol.standard,
            nb_copies=2,
            to_profile=to_profile,  # type: ignore
            from_profile=from_profile,
        )


def test_pure_det_conversion() -> None:
    """coherent state has rank 0 so the trace distance is decreasing in n, . Max is reached for k.n = 0."""
    from_state = CoherentState(amplitude=3)
    to_state = CatState(amplitude=1, parity=False)

    max_rank = 10
    # same parameters for both states
    pars = OptimisationParameters(method=Method.gaussian)

    from_profile = compute_profile(ranks=list(range(max_rank + 1)), target_state=from_state, optim_params=pars)
    to_profile = compute_profile(ranks=list(range(max_rank + 1)), target_state=to_state, optim_params=pars)
    max_dist = max_trace_distance_precision_pure_pure_std(from_profile=from_profile, to_profile=to_profile, nb_copies=2)
    print(f"{max_dist=} and {1 - to_profile.profile[0]}")

    assert isclose(max_dist, 1 - to_profile.profile[0], abs_tol=1e-8)

    max_dist_2 = max_trace_distance_precision(
        protocol=Protocol.standard, nb_copies=2, to_profile=to_profile, from_profile=from_profile
    )
    print(f"{max_dist=} and {max_dist_2}")
    assert isclose(max_dist, max_dist_2)
    # assert False


def test_pure_conversion_post() -> None:
    """test conversion from rank 1 to even cat state amp"""
    # check for 3 copies with Rui's value: 1-f_3(cat) = 1-0.896384 ≅ 0.103616
    true_value = 1 - 0.896384
    to_state = CatState(amplitude=3, parity=False)

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


# TODO add a check for the values if possible?
@pytest.mark.parametrize("prob", [0, 0.0032, 0.249, 0.33337, 0.491, 0.5601, 0.6662, 0.7831, 0.91673645, 1])
def test_mixed_det_conversion(prob: float) -> None:
    """Test if max_distance dispatches correctly to the standard mixed case.

    Parameters
    ----------
    prob : float
        probability parameter in input state
    """

    decomp: PureDecompositionData = (
        (prob, FockState(n=0)),
        (1 - prob, FockState(n=1)),
    )

    from_state = HermitianCVOp(data=decomp)
    from_pars = OptimisationParameters(method=Method.fock, target_cutoff=6)
    to_state = CatState(amplitude=1, parity=False)
    to_pars = OptimisationParameters(method=Method.gaussian)

    max_rank = 10

    from_profile = compute_profile(ranks=list(range(max_rank)), target_state=from_state, optim_params=from_pars)
    to_profile = compute_profile(ranks=list(range(max_rank)), target_state=to_state, optim_params=to_pars)

    max_dist = max_trace_distance_precision_mixed_pure_std(
        from_profile=from_profile, to_profile=to_profile, nb_copies=2
    )

    # assert isclose(max_dist, (1 - to_profile.profile[0] - 2 * (1 - from_profile.profile[0])) ** 2, abs_tol=1e-8)

    max_dist_2 = max_trace_distance_precision(
        protocol=Protocol.standard, nb_copies=2, to_profile=to_profile, from_profile=from_profile
    )
    # print(f"{max_dist=} and {max_dist_2}")
    assert isclose(max_dist, max_dist_2)
    # assert False
