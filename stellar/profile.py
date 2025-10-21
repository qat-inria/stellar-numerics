"""Module related to the computation of the stellar profile
References:
[2] Chabaud et al., https://arxiv.org/pdf/2011.04320 (2020)
[3] Chabaud et al., https://arxiv.org/abs/1907.11009 (2020)
"""

from math import pi as π
from typing import Any

import numpy as np
from scipy.optimize import (
    Bounds,  # noqa: F401
    OptimizeResult,
    basinhopping,
    direct,  # noqa: F401
    minimize,  # noqa: F401
)

from stellar.cvstates import CVState, GaussianState, LCGaussianState
from stellar.params import GaussianParameters
from stellar.gaussian import GaussianOp, Method, Parameterisation

## Notes
# whatever the state we try just care about it's stellar rank?? Eq. 9 https://arxiv.org/abs/2011.04320


def compute_obj_func(
    x: float,
    y: float,
    r: float,
    theta: float,
    max_rank: int,
    target_state: CVState,
    target_cutoff: int | None = None,
    method: str | None = None,
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

    # NOTE add target state cutoff since input will be an abstract state not a concrete statevector

    # NOTE modify here for different target states

    # This is common to all branches
    gauss_params = GaussianParameters(x=x, y=y, r=r, theta=theta)
    g = GaussianOp(gauss_params, method=Method.recursive, param=Parameterisation.Fock)

    # if isinstance(target_state, (GaussianState, LCGaussianState)):
    if method == "g":
        state = g @ target_state
        # now cutoff has to be max_rank
        return np.sum(np.abs(state.get_statevector(cutoff=max_rank).statevector) ** 2)
    # should work for all other states.
    # NOTE Except difficult cases like position, momentum, GKP, cubic phase gate, ...
    # TODO update with more cases when needed
    else:  # TODO check here that cutoff canot be None
        g.build_matrix_fock_basis(bra_cutoff=target_cutoff, ket_cutoff=max_rank)

        # print('here', target_state.get_statevector(cutoff=target_cutoff).dim)
        # vectorized! result is a one dim vector of dim ket_cutoff
        return np.sum(
            np.abs(target_state.get_statevector(cutoff=target_cutoff).statevector.conj() @ g.matrix_fock_basis) ** 2
        )


# need have non custom objects as arguments for scipy minimize
# TODO wrap around optimisation parameters like niter and starting point?
# Yes indeed good workflow. No guarantee so check for several starting points and niter.
# function for that to check convergence?
def compute_sup_fidelity(
    max_rank: int, target_state: CVState, target_cutoff: int | None = None, method: str | None = None
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
    # works but slower than direct which fails on rank = 1, state =|2>
    return basinhopping(
        lambda params: -compute_obj_func(  # type: ignore
            x=params[0],
            y=params[1],
            r=params[2],
            theta=params[3],
            max_rank=max_rank,
            target_state=target_state,
            target_cutoff=target_cutoff,
            method=method,
        ),
        x0=(0.1,) * 4,
        niter=250,
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
