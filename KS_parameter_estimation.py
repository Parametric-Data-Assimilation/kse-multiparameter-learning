# KS_parameter_estimation.py
"""Parameter estimation for the Kuramoto Sivashinsky equation.

Author Benjamin Pachev <benjaminpachev@gmail.com> 2020
"""

import json
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

from KS_order import KS, KSAssim


def fourier_projector(spec, modes=21):
    mod_spec = spec.copy()
    mod_spec[modes:] = 0
    return np.fft.irfft(mod_spec)

def l2_norm(x):
    """Compute the l2 norm, ||x|| = sqrt(sum_{i}(x_i^2))"""
    return (np.sum(np.abs(x)**2)) ** .5


def do_experiment(initial_guess, dt=.01, max_t=10, make_plots=False,
                  timestepper='rk3', modes=21, N=128, warmup_time=10,
                  true_params={}, show_spectrum=False, **kwargs):
    """Run a single full experiment.

    Parameters
    ----------
    initial_guess : dict
        All initial guess parameters to pass to the data assimilator, an object
        of type KS_order.KS.

    dt : float
        Time step

    max_t : float
        Maximum time to integate to

    make_plots : bool
        If true, plot results sequentially.
    """
    true = KS(dt=dt, N=N, timestepper=timestepper, **true_params)
    #The **initial_guess passes all initial guess parameters to the data assimilator
    estimate_params = list(initial_guess.keys())
    projector = partial(fourier_projector, modes=modes)
    assimilator = KSAssim(projector, timestepper=timestepper,
        dt=dt, estimate_params=estimate_params, N=N, **initial_guess, **kwargs)

    for i in range(int(warmup_time/dt)): true.advance()
    print("Warmed up")
    max_n = int(max_t/dt) #10 second similation
    interp_errors = []
    true_errors = []
    param_errors = {p : [] for p in estimate_params}
    # norms of the true and estimated solution
    true_norms = []
    estimate_norms = []
    for n in range(max_n):
        target = fourier_projector(true.xspec)
        projected_state = fourier_projector(assimilator.xspec)
        interp_errors.append(l2_norm(target-projected_state))
        true_errors.append(assimilator.error(true))
        true_norms.append(l2_norm(true.xspec))
        estimate_norms.append(l2_norm(assimilator.xspec))

        for p in param_errors:
            param_errors[p].append(np.abs(getattr(assimilator, p)-getattr(true,p)))
        assimilator.set_target(target)
        assimilator.advance()
        true.advance()

    if make_plots:
        dom = np.arange(max_n) * dt
        plt.semilogy(dom, interp_errors, label="Observed Error")
        plt.semilogy(dom, true_errors, label="True Errors")
        for p in param_errors:
            plt.semilogy(dom, param_errors[p], label=f"{p} Errors")
        # plt.semilogy(dom, true_norms, label="True solution norm")
        # plt.semilogy(dom, estimate_norms, label="Estimated solution norm")
        plt.legend()
        plt.show()

    if show_spectrum:
        assimilator.plot_spectrum()

    return interp_errors, true_errors, param_errors


# =============================================================================
if __name__ == "__main__":
    do_experiment({"lambda2":2,"lambda4":2}, alpha=1, dt=.001, max_t=40, mu=20, order=1, make_plots=True)

