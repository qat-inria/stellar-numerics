from enum import Enum, auto
from typing import TypeGuard, TypeVar, assert_never
import warnings

from stellar.cvstates import HermitianCVOp, PureCVState
from stellar.profile import StellarProfile

# logger = logging.getLogger(__name__)


class Protocol(Enum):
    """Enumeration of the protocols for Gaussian conversion.
    Deterministic and Probabilistic."""

    standard = auto()
    postselected = auto()


# union is empty anyways
# NOTE: is there a better wat to do this?
S = TypeVar("S", PureCVState, HermitianCVOp)
T = TypeVar("T", bound=PureCVState)


def is_stellar_profile_pure(profile: StellarProfile[S]) -> TypeGuard[StellarProfile[PureCVState]]:
    return isinstance(profile.state, PureCVState)


# both have a defined state type but they can be different
# equations will also be different ...
# find a way to filter types.
# Would be nice to have profile.state instead of profile._state
def max_trace_distance_precision(
    protocol: Protocol,
    nb_copies: int,
    to_profile: StellarProfile[T],
    from_profile: StellarProfile[S] | None = None,
    from_rank: int | None = None,
) -> float:  # or None? TODO don't understand this typing issue
    # assume contiguous ranks from 0 to max in all cases

    # flow: will only assess convergence from pure to pure or mixed to pure

    if from_profile is None and from_rank is None:
        raise ValueError("Both `from_profile`and `from_rank` cannot be None.")

    if isinstance(to_profile.state, HermitianCVOp):
        raise ValueError("Cannot assess Gaussian conversion to a mixed state.")

    # pure -> pure case
    # requires both profiles explicitely

    if protocol is Protocol.standard:
        if from_profile is None:
            raise ValueError(
                "`from_profile` cannot be None when assessing Gaussian conversion with a non-postselected protocol."
            )
        if is_stellar_profile_pure(from_profile):
            return max_trace_distance_precision_pure_pure_std(
                from_profile=from_profile, to_profile=to_profile, nb_copies=nb_copies
            )
        assert isinstance(from_profile.state, HermitianCVOp)
        # TODO to be modified here for mixed states
        return 3.0

    elif protocol is Protocol.postselected:
        # TODO check here what happens for mixed states
        if from_profile is not None:
            warnings.warn("`from_profile` is not used in assessing Gaussian conversion with a postselected protocol.")
        if from_rank is None:
            raise ValueError(
                "`from_rank` has to be specified in assessing Gaussian conversion with a postselected protocol"
            )

        return max_trace_distance_precision_pure_pure_post(
            to_profile=to_profile, from_rank=from_rank, nb_copies=nb_copies
        )
    else:
        assert_never(protocol)


# TODO discuss type refinement with Thierry

# not necessarily same pure state...
U = TypeVar("U", bound=PureCVState)
V = TypeVar("V", bound=PureCVState)


def max_trace_distance_precision_pure_pure_std(
    from_profile: StellarProfile[U], to_profile: StellarProfile[V], nb_copies: int
) -> float:
    """finding max trace distance for deterministic conversion between pure states with a fixed number of copies.
    Eq. (34) [HFFC25]"""
    # logger.info("Starting deterministicGaussian conversion analysis...")
    # avoid recomputing this
    max_rank_from = max(from_profile.profile.keys())
    max_rank_to = max(to_profile.profile.keys())

    max_n = min([max_rank_from, max_rank_to // nb_copies])  # floor taken by integer division

    print(f"{max_n=}")
    distance_list: list[float] = []

    # do we need to know the n giving the max?
    for n in range(0, max_n + 1):
        distance_list.append(1 - to_profile.profile[nb_copies * n] - nb_copies * (1 - from_profile.profile[n]))
    print(f"{distance_list=}")
    return max(distance_list)


def max_trace_distance_precision_pure_pure_post(
    to_profile: StellarProfile[V], from_rank: int, nb_copies: int
) -> float:  # or None, error
    """finding max trace distance for postselected conversion between pure states with a fixed number of copies.
    The actual profile of the target state is not required (only the stellar rank) since it is a looser bound see Eq. (35) [HFFC25]."""

    return 1 - to_profile.profile[nb_copies * from_rank]


# TODO add from mixed state converison bounds
