"""Module related to the computation of the stellar profile
References:
[2] Chabaud et al., https://arxiv.org/pdf/2011.04320 (2020)
[3] Chabaud et al., https://arxiv.org/abs/1907.11009 (2020)
"""

from typing import TypeVar
import warnings
from math import pi as π
from typing import Any, Callable
from typing_extensions import assert_never  # for 3.10 compatibility
# for 3.13 from typing import assert_never works

import numpy as np
from scipy.optimize import (
    Bounds,  # noqa: F401
    OptimizeResult,
    basinhopping,
    direct,  # noqa: F401
    minimize,  # noqa: F401
)

from stellar.cvstates import GaussianState, HermitianCVOp, LCGaussianState, Matrix, PureCVState, PureDecompositionData
from stellar.data import StellarProfile
from stellar.gaussian import GaussianOp
from stellar.params import GaussianParameters, Method, OptimisationParameters

## Notes
# whatever the state we try just care about it's stellar rank?? Eq. 9 https://arxiv.org/abs/2011.04320


def compute_obj_func_pure(
    x: float,
    y: float,
    r: float,
    theta: float,
    max_rank: int,
    target_state: PureCVState,
    method: Method,
    target_cutoff: int | None = None,
) -> float:
    """function to be optimized in the pure_state case
    From [2] Thm 1

    Parameters
    ----------
    x : float
        real part of displacement
    y : float
        imaginary part of displacement
    r : float
        modulus of squeezing
    theta : float
        phase of squeezing
    max_rank : int
        stellar rank to compute
    target_state : CVState
        target state
    target_cutoff : int
        cutoff to use when computing the target state's statevector

    Returns
    -------
    float
        the value of the stellar robustness/fidelity

    Raises
    ------
    NotImplementedError
        only works using Fock statevectors so far
    """

    # This is common to all cases
    gauss_params = GaussianParameters(x=x, y=y, r=r, theta=theta)

    # TODO here add support for mixed states. either iterate on the decomp or on the matrix
    # decomp is just a recursive call to this fonction? Can indeed optimise separately
    # but maybe sum everything then optimize

    g = GaussianOp(gauss_params)

    # print(f"working on pur state {target_state}")
    # print(isinstance(target_state, (GaussianState, LCGaussianState)))

    # gaussian method
    if method == Method.gaussian:
        if not isinstance(target_state, (GaussianState, LCGaussianState)):
            raise TypeError(f"The {target_state=} is not a `GaussianState`or `LCGaussianState`.")
        state = g @ target_state
        # now cutoff has to be max_rank
        return -np.sum(np.abs(state.get_statevector(cutoff=max_rank).statevector) ** 2)

    elif method == Method.fock:
        # maybe redundant since checked when calling statevector
        if target_cutoff is None:
            raise ValueError("Cannot compute in the Fock basis if no cutoff is given")
        # common to both pure and composite cases

        # ignore type error since handes at initialisation of OptimisationParameter object
        # cutoff cannot be None
        # this adds new fields
        g.build_matrix_fock_basis(bra_cutoff=target_cutoff, ket_cutoff=max_rank)  # type: ignore

        # vectorized! result is a one dim vector of dim ket_cutoff
        return -np.sum(
            np.abs(target_state.get_statevector(cutoff=target_cutoff).statevector.conj() @ g.matrix_fock_basis) ** 2
        )

    else:
        assert_never(method)


def compute_obj_func_mixed(
    x: float,
    y: float,
    r: float,
    theta: float,
    max_rank: int,
    target_state: HermitianCVOp[PureDecompositionData],
    method: Method,
    target_cutoff: int | None = None,
) -> float:
    """function to be optimized
    From [2] Thm 1

    Parameters
    ----------
    x : float
        real part of displacement
    y : float
        imaginary part of displacement
    r : float
        modulus of squeezing
    theta : float
        phase of squeezing
    max_rank : int
        stellar rank to compute
    target_state : CVState
        target state
    target_cutoff : int
        cutoff to use when computing the target state's statevector

    Returns
    -------
    float
        the value of the stellar robustness/fidelity

    Raises
    ------
    NotImplementedError
        only works using Fock statevectors so far
    """

    # This is common to all cases
    gauss_params = GaussianParameters(x=x, y=y, r=r, theta=theta)

    # TODO here add support for mixed states. either iterate on the decomp or on the matrix
    # decomp is just a recursive call to this fonction? Can indeed optimise separately
    # but maybe sum everything then optimize

    # g = GaussianOp(gauss_params)
    # just compute the sum over the pure state decomposition of the mixed state
    # adjust the method type depending on the state?

    # TODO better way of doing this? No longer the choice here. Whereas can have it in pure state case.
    def method_picker(state: PureCVState) -> Method:
        if isinstance(state, (GaussianState, LCGaussianState)):
            return Method.gaussian

        else:
            return method

    # print(f"{target_state.data=}")

    # TODO: use optim params here too?
    # NOTE no minus sign here since included in the pure state case
    return sum(
        coeff
        * compute_obj_func_pure(
            x=gauss_params.x,
            y=gauss_params.y,
            r=gauss_params.r,
            theta=gauss_params.theta,
            max_rank=max_rank,
            target_state=state,
            method=method_picker(state),
            target_cutoff=target_cutoff,
        )  # mypy yells... silence that due to checking existence of decomposition a level higher
        for coeff, state in target_state.data
    )


# need have non custom objects as arguments for scipy minimize
# TODO wrap around optimisation parameters like niter and starting point?
# Yes indeed good workflow. No guarantee so check for several starting points and niter.
# function for that to check convergence?
def compute_sup_fidelity(
    max_rank: int,
    target_state: PureCVState | HermitianCVOp,
    optim_params: OptimisationParameters,
    # target_cutoff: int | None = None,
    # method: str | None = None,
    # x0: tuple[float, ...] = (0.1,) * 4,
    # niter: int = 250,
    # **kwargs, ## TODO re add kwargs later in OptimizationParameters
) -> OptimizeResult:
    # opt

    # gradient-less? Otherwise numerical gradients? or parameter-shift rule?
    # see default in mathematica
    # multiply by -1 to minimize

    # look at global optimization method instead!
    # like brute force or scipy.optimize.direct
    # or try several starting point in parallel. Performance/cost tradeoff. Dask parallel, pools?
    # return minimize(
    #     lambda params: -compute_obj_func(
    #         x=params[0], y=params[1], r=params[2], theta=params[3], max_rank=max_rank, target_state=target_state
    #     ),
    #     x0=(0,)*4, # (0, 1, 3, .43)
    #     method="COBYLA",
    # )
    # Bounds(lb, ub)
    # return direct(
    #     lambda params: -compute_obj_func(
    #         x=params[0], y=params[1], r=params[2], theta=params[3], max_rank=max_rank, target_state=target_state
    #     ), bounds = Bounds([-3., -3., -3., 0.], [3., 3., 3., 2*pi]),
    #     maxfun=10000  # type: ignore
    # )
    # bad type annotation in scipy.Bounds()

    # specify all bounds and limit squeezing to large but not to large to avoid overflow
    # and not too small squeezing since generic statevector ill-defined
    bounds = Bounds([-np.inf, -np.inf, 1e-5, 0], [np.inf, np.inf, 15, 2 * π])

    # no typing for this object either in scipy or scipy-stubs

    # better to do these checks at this level since will avoid repetition in the logs

    # redundant since only done at `Optimisationparameters` instantiation.
    # if optim_params.method == Method.gaussian and optim_params.target_cutoff is not None:
    #     warnings.warn("`target_cutoff` will be ignored using the `gaussian` method.")

    T = TypeVar("T", bound=PureCVState | HermitianCVOp)

    # cannot name fields here so warning on positional
    def caller(
        state: T, func: Callable[[float, float, float, float, int, T, Method, int | None], float]
    ) -> OptimizeResult:
        # def explicite partial funct
        # params from above scope``
        # HERE
        minimizer_kwargs: dict[str, Any] = {
            # "method": "L-BFGS-B",
            "bounds": bounds,
            "args": (
                max_rank,
                state,
                optim_params.method,
                optim_params.target_cutoff,
            ),
        }

        def partial_func(params, *args):  # no types to avoind problem in basinhoppin?
            return func(params[0], params[1], params[2], params[3], *args)

        return basinhopping(
            partial_func,
            x0=optim_params.x0,
            niter=optim_params.niter,  # default niter = 100
            rng=optim_params.seed,
            # **kwargs,  # other kwargs like rng (seed, or random number generator)
            minimizer_kwargs=minimizer_kwargs,  # type: ignore
        )

    if isinstance(target_state, HermitianCVOp):
        if isinstance(target_state.data, Matrix):
            raise NotImplementedError("Can only optimize on operators built from a pure-state decomposition for now.")

        # know data is not a Matrix so ignore type
        if any(isinstance(state, (GaussianState, LCGaussianState)) for _, state in target_state.data):
            warnings.warn(
                "A `GaussianState` or `LCGaussianState` was detected in the pure-state decomposition. Overriding your `method` choice for this state if it wasn't `gaussian`."
            )
        return caller(target_state, compute_obj_func_mixed)

    # can always duplicate the call directly but looks bad...

    else:
        return caller(target_state, compute_obj_func_pure)
    # TODO: directly use a `GaussParam` object?
    # return basinhopping(
    #     lambda params: -func(
    #         x=params[0],
    #         y=params[1],
    #         r=params[2],
    #         theta=params[3],
    #         max_rank=max_rank,
    #         target_state=target_state,
    #         target_cutoff=optim_params.target_cutoff,
    #         method=optim_params.method,
    #     ),
    #     x0=optim_params.x0,
    #     niter=optim_params.niter,  # default niter = 100
    #     rng=optim_params.seed,
    #     # **kwargs,  # other kwargs like rng (seed, or random number generator)
    #     minimizer_kwargs=minimizer_kwargs,  # type: ignore
    # )


# compute stellar_profile.
# be smart to avoid recomputing the whole matrix for all stellar ranks for given gaussian parameters...
# or do it in parallel?


# TODO reproduce results on cat states in UC's thesis p. 71


# bounds = Bounds([0, -5], [10, 2])

# minimizer_kwargs = {
#     "method": "L-BFGS-B",
#     "bounds": bounds
# }

# x0 = [5, 0]  # Initial guess within the bounds

# result = basinhopping(objective, x0, minimizer_kwargs=minimizer_kwargs)

# the type of SteallarProfile knows about the type of its target state
S = TypeVar("S", bound=PureCVState | HermitianCVOp)


def compute_profile(ranks: list[int], target_state: S, optim_params: OptimisationParameters) -> StellarProfile[S]:
    fidelities: list[float] = []

    for rank in ranks:
        fidelities.append(
            -compute_sup_fidelity(max_rank=rank, target_state=target_state, optim_params=optim_params).fun
        )

    return StellarProfile(state=target_state, ranks=ranks, fidelities=fidelities, optim_params=optim_params)
