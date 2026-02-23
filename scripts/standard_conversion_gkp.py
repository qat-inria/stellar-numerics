"""Script to reproduce Fig. 3 of the paper.
Conversion from odd cat states of varying amplitude to GKP state with Δ = κ = 0.3"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

from stellar.conversion import Protocol, max_trace_distance_precision
from stellar.cvstates import CatState, GKPState
from stellar.data import StellarProfile
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


# --- Global plotting style (paper-ready) ---
rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.7,
        "axes.linewidth": 1.4,
    }
)


if __name__ == "__main__":
    #### file handling
    # directory to save the data
    # will be created if needed
    dir = Path("profile_database/")

    #### parameters
    # amplitudes to compute
    amp_values = [1, 2, 3, 4]  # [sqrt(4)] #[sqrt(2 * i) for i in range(1,6)] # [sqrt(2)]
    niter_values = [400, 400, 400, 500]
    # max number of copies
    max_copies = 15

    # Optimisation parameters. Gaussian method in that case.
    pars = OptimisationParameters(method=Method.gaussian)

    ### To state

    to_state = GKPState()
    # maximum rank achieved for the stellar profile
    max_rank = 30
    niter = 450
    to_fname = f"gkp_0.3_max_rank_{str(max_rank)}_niter_{niter}"
    to_file = dir / Path(to_fname + ".json")

    # loading/computing to_profile

    if to_file.exists() and to_file.is_file():
        logger.info("File exists, loading stellar profile...")

        to_profile = StellarProfile.from_file(filename=to_fname, path=dir)
    else:
        logger.info(f"File doesn't exist, computing stellar profile for {repr(to_state)}...")

        # increase number of iterations for GKP if required
        pars_GKP = OptimisationParameters(method=Method.gaussian, niter=niter)
        to_profile = compute_profile(ranks=list(range(max_rank + 1)), target_state=to_state, optim_params=pars_GKP)
        to_profile.save_to_file(filename=str(to_fname), path=dir)

    # computing the profiles
    tot_data = []
    # amplitude loop
    for k, amp in enumerate(amp_values):
        # use path.name evrywhere for getting file names from path objects
        from_fname = f"cat_odd_{amp:.2f}_max_rank_{str(max_rank)}_niter_{niter_values[k]}"
        from_file = dir / Path(from_fname + ".json")
        # reinitialise data for each new state
        data = []

        # even cats
        from_state = CatState(amplitude=amp, parity=True)

        logger.info("Computing or loading profile...")

        if from_file.exists() and from_file.is_file():
            logger.info("File exists, loading stellar profile...")

            from_profile = StellarProfile.from_file(filename=from_fname, path=dir)
        else:
            logger.info(f"File doesn't exist, computing stellar profile for {repr(from_state)}...")

            from_profile = compute_profile(ranks=list(range(max_rank + 1)), target_state=to_state, optim_params=pars)
            from_profile.save_to_file(filename=from_fname, path=dir)

        logger.info("Finished getting profile...")

        logger.info("Starting conversion analysis...")  # TODO add params here

        # number of copies loop
        for copies in range(1, max_copies + 1):
            logger.info(f"Looping over copies... k={copies}")  # TODO add params here
            data.append(
                max_trace_distance_precision(
                    Protocol.standard, nb_copies=copies, from_profile=from_profile, to_profile=to_profile
                )
            )
        tot_data.append(data)
    logger.info("Finished!")
    print(tot_data)

    #### plotting

    fig, ax = plt.subplots(figsize=(6.5, 4))
    y = range(1, max_copies + 1)
    for i in range(0, len(tot_data)):
        x = tot_data[i]

        ax.step(x, y, where="pre", label=rf"$\alpha={amp_values[i]:.2f}$")

    # Labels with LaTeX
    ax.set_xlabel("Trace distance precision ")  # $\epsilon$
    ax.set_ylabel("Number of copies ")  # $N$

    # Axis limits
    ax.set_xlim(0, 0.41)
    ax.set_ylim(0, 15)  # max(y) + 1

    # Ticks styling
    ax.tick_params(direction="in", length=5, width=1.0)
    # ax.tick_params(which="minor", direction="in", length=3)

    # Remove top/right spines (journal style)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    plt.legend()
    plt.tight_layout()

    # Save for paper
    # plt.savefig("trace_distance_standard_odd_cat_GKP.pdf", bbox_inches="tight")
    # plt.savefig("trace_distance_step_plot.png", dpi=300, bbox_inches="tight")

    plt.show()
