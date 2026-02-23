"""Script to produce stellar profiles."""

from math import e, exp, sqrt

from matplotlib import pyplot as plt, rcParams

from stellar.cvstates import FockState, HermitianCVOp, PureDecompositionData
from stellar.params import Method, OptimisationParameters
from stellar.profile import compute_profile

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


def analytical_stellar_fidelity_mixed_01(prob: float) -> float:
    # check if p is smaller than 1/2
    if prob < 0.5:

        def gp(t: float) -> float:
            # knows prob from scope
            return sqrt((1 + t) ** 3 * (1 - t)) * exp(prob / ((1 - prob) * (1 + t)))

        tstar = 0.25 * (sqrt(9 - 10 * prob / (1 - prob) + prob**2 / (1 - prob) ** 2) + (2 * prob - 1) / (1 - prob))

        return (1 - prob) * gp(tstar) / e

    return prob


if __name__ == "__main__":
    #### file handling
    # directory to save the data
    # will be created if needed
    p_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    niter = 250  # 500
    pars = OptimisationParameters(method=Method.fock, target_cutoff=6, seed=32542)

    f0_data = []
    for prob in p_values:
        # fname = f"mixed_01_prob_{prob:.2f}_max_rank_{str(max_rank)}_niter_{niter}"

        # file = dir / Path(fname + ".json")
        # reinitialise data for each new state

        # mixed state rho_p
        decomp: PureDecompositionData = (
            (prob, FockState(n=0)),
            (1 - prob, FockState(n=1)),
        )
        state = HermitianCVOp(data=decomp)

        profile = compute_profile(ranks=[0], target_state=state, optim_params=pars)
        f0_data.append(profile.profile[0])
    # to_profile.save_to_file(filename=str(fname), path=dir

    analytical_data = [analytical_stellar_fidelity_mixed_01(p) for p in p_values]

    ### plotting
    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(p_values, analytical_data, "-", label="Analytical")
    ax.plot(p_values, f0_data, "+", ms=12, label="Numerical")

    # Labels with LaTeX
    ax.set_xlabel(r"$p$")  # $\epsilon$
    ax.set_ylabel(r"$f_{0, {\rm pure}}^\star(\rho_p)$")  # $N$

    # Axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.1)  # max(y) + 1

    # Ticks styling
    ax.tick_params(direction="in", length=5, width=1.0)
    # ax.tick_params(which="minor", direction="in", length=3)

    # Remove top/right spines (journal style)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    plt.legend()
    plt.tight_layout()

    # Save for paper
    # plt.savefig("mixed_state_check.pdf", bbox_inches="tight")
    # plt.savefig("trace_distance_step_plot.png", dpi=300, bbox_inches="tight")

    plt.show()
