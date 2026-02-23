"""example script for testing the convergence of an
approximate GKP state stellar fidelity with the number of
optimisation iteration
rank = 3 is prone to instability in the standard GKP case.
sols are 0.62246 or 0.599805.
"""

from matplotlib import pyplot as plt, rcParams
from stellar.cvstates import CatState
from stellar.profile import compute_profile
from stellar.params import Method, OptimisationParameters


rcParams.update(
    {
        "text.usetex": True,
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
    tgt_state = CatState(amplitude=5, parity=True)  # GKPState()

    rank = 3

    tot_res = []

    # other parameters for convergence analysis:
    #   - `tol` parameter of the GKP state (how many Gaussians in the decomposition)
    #   - starting point `x0` of the optimization
    #   - ...
    # todo add different seed and starting point to check th evalue
    print("Starting...")
    seeds = [2748, 96762, 42]

    iter_vals = list(range(50, 600, 100))
    # for s in seeds:
    s = 2748
    for x in (0, 1, 0.4725):
        results = []
        for n in iter_vals:
            profile = compute_profile(
                ranks=[rank],
                target_state=tgt_state,
                optim_params=OptimisationParameters(
                    method=Method.gaussian, niter=n, seed=s, x0=(x, -x, x, 0)
                ),  # , seed=s
            )
            results.append(profile.profile[rank])
        tot_res.append(results)
    print("Finished!")
    print(f"The results are: {tot_res=}.")

fig, ax = plt.subplots(figsize=(6.5, 4))
## plotting
ax.tick_params(direction="in", length=5, width=1.0)
for i, s in enumerate(seeds):
    ax.plot(iter_vals, tot_res[i], label=f"seed = {s}")
plt.legend()
plt.tight_layout()
plt.show()
