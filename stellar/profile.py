"""Module related to the computation of the stellar profile
References:
[2] Chabaud et al., https://arxiv.org/pdf/2011.04320 (2020)
[3] Chabaud et al., https://arxiv.org/abs/1907.11009 (2020)
"""

import warnings
from math import pi as π
from typing import Any, assert_never

import numpy as np
from scipy.optimize import (
    Bounds,  # noqa: F401
    OptimizeResult,
    basinhopping,
    direct,  # noqa: F401
    minimize,  # noqa: F401
)

from stellar.cvstates import GaussianState, HermitianCVOp, LCGaussianState, PureCVState
from stellar.gaussian import GaussianOp
from stellar.params import GaussianParameters, Method, OptimisationParameters

## Notes
# whatever the state we try just care about it's stellar rank?? Eq. 9 https://arxiv.org/abs/2011.04320


def compute_obj_func(
    x: float,
    y: float,
    r: float,
    theta: float,
    max_rank: int,
    target_state: PureCVState | HermitianCVOp,
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

    g = GaussianOp(gauss_params)

    # First, check if pure state or operator

    # Pure state case
    if not isinstance(target_state, HermitianCVOp):
        # gaussian method
        if method == Method.gaussian:

            if not isinstance(target_state, (GaussianState, LCGaussianState)):
                raise TypeError(f"The {target_state=} is not a `GaussianState`or `LCGaussianState`.")
            state = g @ target_state
            # now cutoff has to be max_rank
            return np.sum(np.abs(state.get_statevector(cutoff=max_rank).statevector) ** 2)

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
            return np.sum(
                np.abs(target_state.get_statevector(cutoff=target_cutoff).statevector.conj() @ g.matrix_fock_basis) ** 2
            )

        else:
            assert_never(method)

    # Operator case (density matrix or general hermitian operator)

    # sure to have a decomposition since checked at the level above
    elif isinstance(target_state, HermitianCVOp):

        # just compute the sum over the pure state decomposition of the mixed state
        # adjust the method type depending on the state?

        # TODO better way of doing this? No longer the choice here. Whereas can have it in pure state case.
        def method_picker(state: PureCVState) -> Method:
            if isinstance(state, (GaussianState, LCGaussianState)):
                return Method.gaussian

            else:
                return method

        # TODO: use optim params here too?
        return sum(
            coeff
            * compute_obj_func(
                x=gauss_params.x,
                y=gauss_params.y,
                r=gauss_params.r,
                theta=gauss_params.theta,
                max_rank=max_rank,
                target_state=state,
                method=method_picker(state),
                target_cutoff=target_cutoff,
            ) #mypy yells... silence that due to checking existence of decomposition a level higher
            for coeff, state in target_state.decomposition # type: ignore
        )
# safe since decomposition existence checked at the level above
    else:
        assert False
    # no need for else statement, mypy statically checks that the Method Enum is exhausted
    # or assert_never(method)?


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
    minimizer_kwargs: dict[str, Any] = {
        # "method": "L-BFGS-B",
        "bounds": bounds
    }


    # better to do these checks at this level since will avoid repetition in the logs

    # redundant since only done at `Optimisationparameters` instantiation.
    # if optim_params.method == Method.gaussian and optim_params.target_cutoff is not None:
    #     warnings.warn("`target_cutoff` will be ignored using the `gaussian` method.")

    if isinstance(target_state, HermitianCVOp) and target_state.decomposition is not None:

        if target_state.decomposition is None:
            raise NotImplementedError("Can only optimize on operators built from a pure-state decomposition for now.")

        if any(isinstance(state, (GaussianState, LCGaussianState)) for _, state in target_state.decomposition):
                warnings.warn(
                    "A `GaussianState` or `LCGaussianState` was detected in the pure-state decomposition. Overriding your `method` choice for this state if it wasn't `gaussian`."
                )

    # TODO: directly use a `GaussParam` object.
    return basinhopping(
        lambda params: -compute_obj_func(  # type: ignore
            x=params[0],
            y=params[1],
            r=params[2],
            theta=params[3],
            max_rank=max_rank,
            target_state=target_state,
            target_cutoff=optim_params.target_cutoff,
            method=optim_params.method,
        ),
        x0=optim_params.x0,
        niter=optim_params.niter,  # default niter = 100
        rng=optim_params.seed,
        # **kwargs,  # other kwargs like rng (seed, or random number generator)
        minimizer_kwargs=minimizer_kwargs,  # type: ignore
    )


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
