"""example script for testing the convergence of an
approximate GKP state stellar fidelity with the number of
optimisation iteration
rank = 3 is prone to instability in the standard GKP case.
sols are 0.62246 or 0.599805.
"""

import logging
from stellar.cvstates import GKPState
from stellar.profile import compute_sup_fidelity

# logger = logging.getLogger(__name__)
# logging.basicConfig(filename='gkp.log', encoding='utf-8', level=logging.DEBUG)
# logger.debug('This message should go to the log file')
# logger.info('So should this')
# logger.warning('And this, too')
# logger.error('And non-ASCII stuff, too, like Øresund and Malmö')

# logger.info("Starting GKP script...")

if __name__ == "__main__":

    tgt_state = GKPState()

    rank = 3

    results = []

    # other parameters for convergence analysis:
    #   - `tol` parameter of the GKP state (how many Gaussians in the decomposition)
    #   - starting point `x0` of the optimization
    #   - ...
    # todo add different seed and starting point to check th evalue
    print("Starting...")
    for n in range(100, 600, 100):

        res = compute_sup_fidelity(max_rank=rank, target_state=tgt_state, method="g", niter=n)
        results.append(-res.fun)

    print("Finished!")
    print(f"The results are: {results=}.")


