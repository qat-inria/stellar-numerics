import warnings
from enum import Enum, auto
from typing import TypeVar, assert_never

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
    """Bound conversion error using stellar profiles and a protocol.

    Parameters
    ----------
    protocol : Protocol
        Deterministic (``standard``) or postselected conversion protocol.
    nb_copies : int
        Number of input copies used in the conversion.
    to_profile : StellarProfile
        Profile of the pure target state.
    from_profile : StellarProfile or None, optional
        Profile of the input state. Required for the standard protocol and
        ignored for the postselected protocol.
    from_rank : int or None, optional
        Input stellar rank. Required for the postselected protocol.

    Returns
    -------
    float
        Maximum trace-distance precision bound for the selected protocol.

    Raises
    ------
    ValueError
        If the required input profile or rank is missing.
    TypeError
        If the target profile represents a mixed state.

    Warns
    -----
    UserWarning
        If an input profile is supplied for the postselected protocol.
    """
    # assume contiguous ranks from 0 to max in all cases

    # flow: will only assess convergence from pure to pure or mixed to pure

    if from_profile is None and from_rank is None:
        raise ValueError("Both `from_profile`and `from_rank` cannot be None.")

    if isinstance(to_profile.state, HermitianCVOp):
        raise TypeError("Cannot assess Gaussian conversion to a mixed state.")

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
            # print(("target state is mixed"))
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
    """Compute the deterministic conversion bound for pure states.

    Parameters
    ----------
    from_profile : StellarProfile
        Stellar profile of the pure input state.
    to_profile : StellarProfile
        Stellar profile of the pure target state.
    nb_copies : int
        Number of input copies used in the conversion.

    Returns
    -------
    float
        Maximum trace-distance bound evaluated over the ranks available in
        both profiles.

    Notes
    -----
    Implements Eq. (36) of [HGFFC25]. The profiles are expected to contain
    contiguous ranks from zero through their maximum rank.
    """
    # logger.info("Starting deterministicGaussian conversion analysis...")
    # avoid recomputing this
    max_rank_from = max(from_profile.profile.keys())
    max_rank_to = max(to_profile.profile.keys())

    max_n = min([max_rank_from, max_rank_to // nb_copies])  # floor taken by integer division

    # print(f"{max_n=}")
    distance_list: list[float] = []

    for n in range(max_n + 1):
        distance_list.append(1 - to_profile.profile[nb_copies * n] - nb_copies * (1 - from_profile.profile[n]))
    # print(f"{distance_list=}")
    max_dist = max(distance_list)
    print(f"index of maximum is {distance_list.index(max_dist)}")
    return max_dist


def max_trace_distance_precision_mixed_pure_std(
    from_profile: StellarProfile[W], to_profile: StellarProfile[V], nb_copies: int
) -> float:
    """Compute the deterministic conversion bound from a mixed to pure state.

    Parameters
    ----------
    from_profile : StellarProfile
        Stellar profile of the mixed input state.
    to_profile : StellarProfile
        Stellar profile of the pure target state.
    nb_copies : int
        Number of input copies used in the conversion.

    Returns
    -------
    float
        Squared maximum bound evaluated over the ranks available in both
        profiles.

    Notes
    -----
    Implements Eq. (38) of [HGFFC25]. The profiles are expected to contain
    contiguous ranks from zero through their maximum rank.
    """
    max_rank_from = max(from_profile.profile.keys())
    max_rank_to = max(to_profile.profile.keys())

    max_n = min([max_rank_from, max_rank_to // nb_copies])  # floor taken by integer division

    print(f"{max_n=}")
    sqrt_distance_list: list[float] = []
    # print("we are here")
    # TODO add logic here to deal when there are not large negative numbers?
    for n in range(max_n + 1):
        print(1 - to_profile.profile[nb_copies * n] - nb_copies * (1 - from_profile.profile[n]))
        sqrt_distance_list.append(1 - to_profile.profile[nb_copies * n] - nb_copies * (1 - from_profile.profile[n]))
    print(f"{sqrt_distance_list=} and {max(sqrt_distance_list)}")
    return (max(sqrt_distance_list)) ** 2


def max_trace_distance_precision_pure_pure_post(
    to_profile: StellarProfile[V], from_rank: int, nb_copies: int
) -> float:  # or None, error
    """Compute the postselected conversion bound for pure states.

    Parameters
    ----------
    to_profile : StellarProfile
        Stellar profile of the pure target state.
    from_rank : int
        Stellar rank of the input state.
    nb_copies : int
        Number of input copies used in the conversion.

    Returns
    -------
    float
        Trace-distance bound evaluated at target rank
        ``nb_copies * from_rank``.

    Notes
    -----
    Implements the looser bound in Eq. (37) of [HFFC25]; the input state's
    profile is not needed.
    """

    return 1 - to_profile.profile[nb_copies * from_rank]


def max_trace_distance_precision_mixed_pure_post(
    to_profile: StellarProfile[W], from_rank: int, nb_copies: int
) -> float:  # or None, error
    """Compute the postselected conversion bound for a mixed input.

    Parameters
    ----------
    to_profile : StellarProfile
        Profile used to evaluate the bound at the target rank.
    from_rank : int
        Stellar rank of the input state.
    nb_copies : int
        Number of input copies used in the conversion.

    Returns
    -------
    float
        Squared trace-distance bound evaluated at target rank
        ``nb_copies * from_rank``.

    Notes
    -----
    Implements the looser bound in Eq. (39) of [HFFC25].
    """

    return (1 - to_profile.profile[nb_copies * from_rank]) ** 2
