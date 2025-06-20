"""Module related to the computation of the stellar profile
References:
[2] Chabaud et al., https://arxiv.org/pdf/2011.04320 (2020)
[3] Chabaud et al., https://arxiv.org/abs/1907.11009 (2020)
"""

import numpy as np
from scipy.optimize import (
    minimize,
    direct,
    basinhopping,
    OptimizeResult,
    Bounds,
)  

from stellar.gaussian import GaussianOp, GaussianParameters, Method, Parameterisation
from stellar.states import StateFockBasis

## Notes
# whatever the state we try just care about it's stellar rank?? Eq. 9 https://arxiv.org/abs/2011.04320


def compute_obj_func(x: float, y: float, r: float, theta: float, max_rank: int, target_state: StateFockBasis) -> float:
    """function to be optimized
    From [2] Thm 1
    """
    gauss_params = GaussianParameters(x=x, y=y, r=r, theta=theta)
    g = GaussianOp(gauss_params, method=Method.recursive, param=Parameterisation.Fock)
    g.build_matrix_fock_basis(bra_cutoff=target_state.dim, ket_cutoff=max_rank + 1)

    # vectorized! result is a one dim vector of dim ket_cutoff
    return np.sum(np.abs(target_state.statevector.conj() @ g.matrix_fock_basis) ** 2)


# need have non custom objects as arguments for scipy minimize
def compute_sup_fidelity(max_rank: int, target_state: StateFockBasis) -> OptimizeResult:
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

    # works but slower than direct which fails on rank = 1, state =|2>
    return basinhopping(
        lambda params: -compute_obj_func( # type: ignore
            x=params[0], y=params[1], r=params[2], theta=params[3], max_rank=max_rank, target_state=target_state
        ),
        x0=(0,) * 4,
    ) 


# compute stellar_profile.
# be smart to avoid recomputing the whole matrix for all stellar ranks for given gaussian parameters...
# or do it in parallel?
