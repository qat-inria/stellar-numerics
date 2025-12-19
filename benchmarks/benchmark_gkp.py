import cProfile
import time
from stellar.cvstates import GKPState
from stellar.params import Method, OptimisationParameters
from stellar.profile import compute_profile


def get_gkp_profile() -> None:

    state = GKPState(delta=0.3, kappa=0.3)  # more terms (11 vs 4) than Δ = κ = 0.3

    max_rank = 10  # why 20 profile file?
    # same parameters for both states
    pars = OptimisationParameters(method=Method.gaussian, niter=350)
    # init_time = time.time()
    compute_profile(ranks=list(range(max_rank)), target_state=state, optim_params=pars)
    # profile_time = time.time()
    # print(f"profile time {profile_time - init_time}")

if __name__ == '__main__':
    # cProfile.run('get_gkp_profile()', sort='ncalls')
    get_gkp_profile()