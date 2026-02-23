"""Script to produce stellar profiles."""

import logging
import sys
from pathlib import Path
import warnings


from stellar.cvstates import CatState
from stellar.params import Method, OptimisationParameters
from stellar.profile import compute_profile

# Create logger
logger = logging.getLogger("console_logger")
logger.setLevel(logging.DEBUG)

# Create and configure console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Set formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)


if __name__ == "__main__":
    #### file handling
    # directory to save the data
    # will be created if needed
    dir = Path("profile_database/")

    # maximum rank achieved for the stellar profile
    max_rank = 30
    niter = 500
    # Optimisation parameters. Gaussian method. 350 large for cats.
    # pars = OptimisationParameters(method=Method.fock, niter=niter, seed=652887, target_cutoff=6)
    pars = OptimisationParameters(method=Method.gaussian, niter=niter, seed=825499)  # , target_cutoff = 4
    # 2484211
    # pars = OptimisationParameters(method=Method.fock, target_cutoff=6, seed=32542)

    # fname = f"gkp_0.3_max_rank_{str(max_rank)}_niter_{niter}"
    amp = 2
    fname = f"cat_odd_{amp:.2f}_max_rank_{str(max_rank)}_niter_{niter}"
    # fname = f"fock_3_max_rank_{str(max_rank)}_niter_{niter}"
    # fname = f"bin_even_N_2_S_1_max_rank_{str(max_rank)}_niter_{niter}"
    # prob = 0.9
    # fname = f"mixed_01_prob_{prob:.2f}_max_rank_{str(max_rank)}_niter_{niter}"

    file = dir / Path(fname + ".json")
    # reinitialise data for each new state

    # # default GKP state (4 gaussians)
    # to_state = GKPState(delta=0.3, kappa=0.3)

    # cats
    to_state = CatState(amplitude=amp, parity=True)
    # to_state = BinomialState(N=2, S=1, parity=False)# FockState(n=3)
    # mixed state rho_p
    # decomp: PureDecompositionData = (
    #     (prob, FockState(n=0)),
    #     (1 - prob, FockState(n=1)),
    # )

    # to_state = HermitianCVOp(data=decomp)

    logger.info(f"Computing the profile for {repr(to_state)}...")

    if file.exists() and file.is_file():
        warnings.warn("File already exists, this will overwrite it.")

    logger.info("File doesn't exist, computing stellar profile...")

    to_profile = compute_profile(ranks=list(range(max_rank + 1)), target_state=to_state, optim_params=pars)
    to_profile.save_to_file(filename=str(fname), path=dir)

    logger.info("Finished computing profile!")
