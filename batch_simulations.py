# batch_simulations.py
"""Manage several simulations in batch form."""

import os
import sys
import json
import shutil
import traceback
import itertools
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import multiprocessing as mp
from functools import partial
from scipy.interpolate import interp1d

from KS_order import KS, KSAssim


def fourier_projector(spec, modes=21):
    mod_spec = spec.copy()
    mod_spec[modes:] = 0
    return np.fft.irfft(mod_spec)


def pointwise_projector(spec, interp_points, domain, interpolation="cubic"):
    """Project the data based on point-wise observations.

    Parameters
    ----------
    spec : ndarray
        The true state in Fourier space
    interp_points : ndarray
        The points at which we observe the solution.
    domain : ndarray
        The problem domain (important for some interpolation methods)
    """
    x = np.fft.irfft(spec)
    observations = interp1d(domain, x, kind="linear")(interp_points)
    interpolator = interp1d(interp_points, observations,
                            kind=interpolation, fill_value="extrapolate")
    return interpolator(domain)


def l2_norm(x):
    return (np.sum(np.abs(x)**2)) ** .5


# This needs to be a function, not a class method,
# so it can be used with the multiprocessing library.
def run_simulation(initial_guess={"lambda2": 2}, mu=1, dt=.01,
                   alpha=None, max_t=10, modes=21, alpha_scale=None,
                   mu_scale=None, order=2, timestepper="rk3",
                   lambda2=1, N=512, start_xspec=None,
                   start_x=None, pointwise_interpolation=None, **kwargs):
    # Initialize true solution from common starting point
    true = KS(dt=dt, N=N,
              lambda2=lambda2, timestepper=timestepper, **kwargs)
    if start_x is not None and start_xspec is not None:
        true.xspec = start_xspec.copy()
        true.x = start_x.copy()

    estimate_params = list(initial_guess.keys())

    if pointwise_interpolation is not None:
        domain = true.get_domain()
        num_interpolation_points = modes
        # Get num_interpolation_points pts spaced evenly across the grid
        # spacing = N // num_interpolation_points
        # inds = np.arange(0, N, spacing)[:num_interpolation_points]
        interp_points = np.linspace(domain[0], domain[-1],
                                    num_interpolation_points)
        projector = partial(pointwise_projector,
                            interp_points=interp_points, domain=domain,
                            interpolation=pointwise_interpolation)
    else:
        projector = partial(fourier_projector, modes=modes)
    kwargs = {k:v for k,v in kwargs.items() if k not in initial_guess}
    assimilator = KSAssim(projector, N=N, mu=mu, alpha=alpha,
                          dt=dt, timestepper=timestepper, order=order,
                          estimate_params=estimate_params,
                          **initial_guess, **kwargs)
    max_n = int(max_t/dt)
    interp_errors = []
    true_errors = []
    param_errors = {p: [] for p in estimate_params}

    for n in range(max_n):
        target = projector(true.xspec)
        projected_state = projector(assimilator.xspec)
        interp_errors.append(l2_norm(target-projected_state))
        true_errors.append(assimilator.error(true))

        for p in param_errors:
            param_errors[p].append(
                np.abs(getattr(assimilator, p) - getattr(true,p))
            )
        assimilator.set_target(target)
        assimilator.advance()
        true.advance()

    result = {p: np.array(arr) for p,arr in param_errors.items()}
    result.update({
        "interp_errors": np.array(interp_errors),
        "true_errors": np.array(true_errors)}
    )
    return result


def simulation_wrapper(params, **kwargs):
    try:
        return run_simulation(**params, **kwargs)
    except Exception:
        message = traceback.format_exc()
        print(params, message)
        return None


class BatchSimulator:
    """A class to run batch simulations of KSE."""
    def __init__(self, outdir, lambda2=1, N=256,
                 warmup_time=10, warmup_dt=1e-4, overwrite=False):
        """Run normal KSE for 10 seconds to get into the chaotic realm."""

        warmup = KS(lambda2=lambda2, N=N, dt=warmup_dt)
        for i in range(int(warmup_time/warmup_dt)):
            warmup.advance()

        self.start_x = warmup.x
        self.start_xspec = warmup.xspec
        self.outdir = outdir
        self.N = N
        self.lambda2 = lambda2
        # Make the directory, ensuring uniqueness.
        if os.path.exists(outdir):
            if not overwrite:
                ans = input(f"Directory {outdir} already exists!"
                            " Remove [Y/N]? ")
                if ans.strip().lower() != "y":
                    print("Aborting!")
                    sys.exit(0)
            shutil.rmtree(outdir)
        os.makedirs(outdir)
        print("Initialized the chaotic initial state.")

    def _expand_scales(self, params):
        if "dt" in params:
            if "mu_scale" in params:
                params["mu"] = params["mu_scale"] / params["dt"]
            if "alpha_scale" in params:
                params["alpha"] = params["alpha_scale"] / params["dt"]

    def get_param_list(self, base_params, ranges={}, grid=True):
        if not len(ranges):
            raise RuntimeWarning("No parameter ranges specified "
                                 "(nothing to do!)")
            return

        param_list = []
        params_to_vary = list(ranges.keys())
        if "mu_scale" in base_params and "mu" in ranges:
            raise ValueError("Cannot both set mu_scale and vary mu!")
        if "alpha_scale" in base_params and "alpha" in ranges:
            raise ValueError("Cannot both set alpha_scale and vary alpha!")

        if grid is True:
            for choice in itertools.product(
                *[ranges[p] for p in params_to_vary]
            ):
                input_params = deepcopy(base_params)
                for c, p in zip(choice, params_to_vary):
                    input_params[p] = c

                self._expand_scales(input_params)
                param_list.append(input_params)
        else:
            # vary one parameter at a time - no grid search
            for p, param_vals in ranges.items():
                for val in param_vals:
                    input_params = deepcopy(base_params)
                    input_params[p] = val
                    self._expand_scales(input_params)
                    param_list.append(input_params)

        return param_list

    def run_batch(self, base_params, ranges={}, grid=True, n_jobs=None):
        param_list = self.get_param_list(base_params, ranges=ranges, grid=grid)
        print(param_list)
        self.run_simulations_low(param_list, n_jobs=n_jobs)

    def run_simulations_low(self, param_list, n_jobs=None):
        index = []
        index_file = f"{self.outdir}/index.json"
        if n_jobs is None:
            # Run at 75% capacity by default
            n_jobs = (3*mp.cpu_count() // 4)

        n_simulations = len(param_list)
        processes = min(max(1, n_jobs), n_simulations)
        pool = mp.Pool(processes=processes)
        func = partial(simulation_wrapper,
                       start_xspec=self.start_xspec,
                       start_x=self.start_x,
                       N=self.N,
                       lambda2=self.lambda2)

        i = 0
        for result in tqdm(
            pool.imap(func=func, iterable=param_list), total=n_simulations
        ):
            params = param_list[i]
            if result is not None:
                filename = f"{self.outdir}/results_{i}.npz"
                np.savez(filename, **result)
                index.append({
                    "succeded": True,
                    "params": params,
                    "filename": filename})
            else:
                index.append({
                    "succeded": False,
                    "params": params,
                    "error":""})
            i += 1
            if (i+1) % 10 == 0:
                print(f"Completed {i+1} of {len(param_list)} simulations,"
                      " saving index file.")
                with open(index_file, "w") as fp:
                    json.dump(index, fp)

        print(f"Completed all {len(param_list)} simulations,"
              " saving index file.")
        with open(index_file, "w") as fp:
            json.dump(index, fp)


# =============================================================================
if __name__ == "__main__":
    sim = BatchSimulator("test", warmup_time=0.001)
    sim.run_batch({"initial_guess": {"lambda2": 2}},
                  ranges={
                      "mu_scale": [.8,1.8],
                      "alpha_scale": [.01, .1],
                      "dt":[1e-2, 1e-3, 1e-4]})
