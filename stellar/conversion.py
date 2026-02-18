from enum import Enum, auto
from typing import TypeVar, assert_never
import warnings

from stellar.cvstates import HermitianCVOp, PureCVState
from stellar.profile import StellarProfile

# logger = logging.getLogger(__name__)


class Protocol(Enum):
    """Enumeration of the protocols for Gaussian conversion.
    Deterministic and Probabilistic."""

    standard = auto()
    postselected = auto()


# both have a defined state type but they can be different
# equations will also be different ...
# find a way to filter types.
# Would be nice to have profile.state instead of profile._state
def max_trace_distance_precision(
    protocol: Protocol,
    nb_copies: int,
    to_profile: StellarProfile[PureCVState],
    from_profile: StellarProfile[PureCVState | HermitianCVOp] | None = None,
    from_rank: int | None = None,
) -> float:  # or None? TODO don't understand this typing issue
    # assume contiguous ranks from 0 to max in all cases

    # flow: will only assess convergence from pure to pure or mixed to pure

    if from_profile is None and from_rank is None:
        raise ValueError("Both `from_profile`and `from_rank` cannot be None.")

    if isinstance(to_profile.state, HermitianCVOp):
        raise ValueError("Cannot assess Gaussian conversion to a mixed state.")

    match protocol:
        # standard protocol
        # requires both profiles explicitely
        case Protocol.standard:
            if from_profile is None:
                raise ValueError(
                    "`from_profile` cannot be None when assessing Gaussian conversion with a non-postselected protocol."
                )
            # pure -> pure case
            if isinstance(from_profile.state, PureCVState):
                # In this branch, the typer knows that `from_profile.state` has type `PureCVState`,
                # therefore `from_profile.replace(from_profile.state)` has type `StellarProfile[PureCVState]`.
                return max_trace_distance_precision_pure_pure_std(
                    from_profile=from_profile.replace(from_profile.state), to_profile=to_profile, nb_copies=nb_copies
                )
            # mixed -> pure case
            print(("state is mixed"))
            assert isinstance(from_profile.state, HermitianCVOp)
            return max_trace_distance_precision_mixed_pure_std(
                from_profile=from_profile.replace(from_profile.state), to_profile=to_profile, nb_copies=nb_copies
            )
        # postselected protocol
        case Protocol.postselected:
            # TODO implement here the logic for postselected protocol for mixed states
            # can always take the square root manually if needed
            if from_profile is not None:
                warnings.warn(
                    "`from_profile` is not used in assessing Gaussian conversion with a postselected protocol."
                )
            if from_rank is None:
                raise ValueError(
                    "`from_rank` has to be specified in assessing Gaussian conversion with a postselected protocol"
                )
            return max_trace_distance_precision_pure_pure_post(
                to_profile=to_profile, from_rank=from_rank, nb_copies=nb_copies
            )

        case _:
            assert_never(protocol)


# not necessarily same pure state
U = TypeVar("U", bound=PureCVState)
V = TypeVar("V", bound=PureCVState)
W = TypeVar("W", bound=HermitianCVOp)


def max_trace_distance_precision_pure_pure_std(
    from_profile: StellarProfile[U], to_profile: StellarProfile[V], nb_copies: int
) -> float:
    """finding max trace distance for deterministic conversion between pure states with a fixed number of copies.
    Eq. (36) [HGFFC25]"""
    # logger.info("Starting deterministicGaussian conversion analysis...")
    # avoid recomputing this
    max_rank_from = max(from_profile.profile.keys())
    max_rank_to = max(to_profile.profile.keys())

    max_n = min([max_rank_from, max_rank_to // nb_copies])  # floor taken by integer division

    # print(f"{max_n=}")
    distance_list: list[float] = []

    for n in range(0, max_n + 1):
        distance_list.append(1 - to_profile.profile[nb_copies * n] - nb_copies * (1 - from_profile.profile[n]))
    print(f"{distance_list=}")
    return max(distance_list)


def max_trace_distance_precision_mixed_pure_std(
    from_profile: StellarProfile[W], to_profile: StellarProfile[V], nb_copies: int
) -> float:
    """finding max trace distance for deterministic conversion from a mixed state to a pure state with a fixed number of copies.
    Eq. (38) [HGFFC25]"""
    max_rank_from = max(from_profile.profile.keys())
    max_rank_to = max(to_profile.profile.keys())

    max_n = min([max_rank_from, max_rank_to // nb_copies])  # floor taken by integer division

    # print(f"{max_n=}")
    distance_list: list[float] = []

    for n in range(0, max_n + 1):
        distance_list.append((1 - to_profile.profile[nb_copies * n] - nb_copies * (1 - from_profile.profile[n])) ** 2)
    print(f"{distance_list=}")
    return max(distance_list)


def max_trace_distance_precision_pure_pure_post(
    to_profile: StellarProfile[V], from_rank: int, nb_copies: int
) -> float:  # or None, error
    """finding max trace distance for postselected conversion between pure states with a fixed number of copies.
    The actual profile of the target state is not required (only the stellar rank) since it is a looser bound see Eq. (37) [HFFC25]."""

    return 1 - to_profile.profile[nb_copies * from_rank]


def max_trace_distance_precision_mixed_pure_post(
    to_profile: StellarProfile[W], from_rank: int, nb_copies: int
) -> float:  # or None, error
    """finding max trace distance for postselected conversion from a mixed state to a pure state with a fixed number of copies.
    The actual profile of the target state is not required (only the stellar rank) since it is a looser bound see Eq. (39) [HFFC25]."""

    return (1 - to_profile.profile[nb_copies * from_rank]) ** 2
