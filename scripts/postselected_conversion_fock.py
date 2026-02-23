"""Script to reproduce Fig. 4 of the paper;
Conversion from any stellar rank 1 state to even cat states of varying amplitude"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

from stellar.conversion import Protocol, max_trace_distance_precision
from stellar.cvstates import FockState
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
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.2,
    }
)


if __name__ == "__main__":
    #### file handling
    # directory to save the data
    # will be created if needed
    dir = Path("profile_database/")

    #### parameters
    # amplitudes to compute
    # amp_values = [sqrt(2 * i) for i in range(1,6)] # [sqrt(2)]

    # max number of copies
    max_copies = 10

    # maximum rank achieved for the stellar profile
    max_rank = 10

    # convert from any stellar rank 1 input state
    from_rank = 1

    # Optimisation parameters. Gaussian method. Rest is default.
    pars = OptimisationParameters(method=Method.fock, target_cutoff=5)

    # computing the profiles
    tot_data = []
    # amplitude loop

    fname = f"fock_2_max_rank_{str(max_rank)}_niter_200"
    file = dir / Path(fname + ".json")
    # reinitialise data for each new state
    data = []

    # even cats
    to_state = FockState(n=2)

    logger.info("Computing or loading profile...")

    if file.exists() and file.is_file():
        logger.info("File exists, loading stellar profile...")

        to_profile = StellarProfile.from_file(filename=fname, path=dir)
    else:
        logger.info("File doesn't exist, computing stellar profile...")

        to_profile = compute_profile(ranks=list(range(max_rank + 1)), target_state=to_state, optim_params=pars)
        to_profile.save_to_file(filename=str(fname), path=dir)

        logger.info("Finished getting profile...")

        logger.info("Starting conversion analysis...")  # TODO add params here

        # number of copies loop
    for copies in range(1, max_copies + 1):
        data.append(
            max_trace_distance_precision(
                Protocol.postselected, nb_copies=copies, to_profile=to_profile, from_rank=from_rank
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

    ax.step(x, y, where="pre")

# Labels with LaTeX
ax.set_xlabel("Trace distance precision ")  # $\epsilon$
ax.set_ylabel("Number of copies ")  # $N$

# Axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, max(y) + 1)

# Ticks styling
ax.tick_params(direction="in", length=5, width=1.0)
# ax.tick_params(which="minor", direction="in", length=3)

# Remove top/right spines (journal style)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
plt.legend()
plt.tight_layout()

# Save for paper
# plt.savefig("trace_distance_postselected_even_cat.pdf", bbox_inches="tight")
# plt.savefig("trace_distance_step_plot.png", dpi=300, bbox_inches="tight")

plt.show()
