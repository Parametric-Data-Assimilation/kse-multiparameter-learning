# paper_simulations.py
"""Source code for the simulations used in the publication."""
import argparse
import itertools

from batch_simulations import BatchSimulator
from simulation_results import SimulationResults


func_dict = {}


def option(func):
    """Add the function to the global list of options"""
    func_dict[func.__name__] = func
    return func


@option
def probe_order(dirname):
    sim = BatchSimulator(dirname, N=512, lambda2=1, overwrite=True)
    base_params = {
        "alpha": 1,
        "mu_scale": 1.8,
        "max_t": 60,
        "modes": 21,
        "timestepper":"rk4"
    }
    base_params["initial_guess"] = {"lambda2": 2}
    ranges = {
        "dt": [1e-2, 1e-3, 5e-3, 1e-4, 5e-4],
        "order": [1,2,3]
    }
    sim.run_batch(base_params, ranges=ranges)


@option
def multiparam_combinations(dirname):
    sim = BatchSimulator(dirname, N=256, lambda2=1, overwrite=True)
    base_params = {
        "alpha": 1,
        "mu_scale": 1.8,
        "max_t": 80,
        "modes": 21,
        "order": 2,
        "dt": 1e-3
    }
    # lambda4 is the coefficient by the fourth derivative u_xxxx,
    # lambda2 is the coefficient by the second derivative term u_xx.
    estimate_params = [
        "lambda2",
        "lambda4",
        "nonlinear_coeff",
        "lambda1",
        "lambda3"
    ]
    initial_guesses = []
    for included in itertools.product([False, True], repeat=5):
        # Skip the boring case of nothing to estimate.
        if not any(included):
            continue
        initial_guess = {}
        for is_included, param_name in zip(included, estimate_params):
            if is_included:
                initial_guess[param_name] = 2
        initial_guesses.append(initial_guess)
    sim.run_batch(base_params, ranges={"initial_guess": initial_guesses})


@option
def high_spatial_res(dirname):
    sim = BatchSimulator(dirname, N=1024, lambda2=1, overwrite=True)
    base_params = {
        "alpha": 1,
        "mu_scale": 1.8,
        "order": 3,
        "max_t": 45,
        "modes": 21,
        "timestepper":"rk4"
    }
    base_params["initial_guess"] = {"lambda2": 2}
    sim.run_batch(base_params, ranges={"dt": [2.5e-4], "modes": [42, 84, 168]})
    results = SimulationResults(dirname)
    df = results.get_summary()
    print(df)
    print(df[["dt", "modes", "lambda2_error"]])


@option
def explore_alpha(dirname):
    sim = BatchSimulator(dirname, N=512, lambda2=1, overwrite=True)
    base_params = {
        "alpha": 1,
        "mu_scale": 1.8,
        "order": 3,
        "max_t": 60,
        "dt": 1e-3,
        "modes": 21,
        "timestepper": "rk4"
    }
    ranges = {
        "initial_guess": [
            {"lambda2": 2},
            {"lambda2": 2, "lambda4": 2, "nonlinear_coeff": 2}
        ],
        "alpha": [.5, 1, 10, 100, None]
    }
    sim.run_batch(base_params, ranges=ranges)


@option
def mu_alpha_convergence(dirname):
    """Vary mu_scale and alpha - and see how the convergence rate changes.
    """
    sim = BatchSimulator(dirname, N=512, lambda2=1, overwrite=True)
    base_params = {
        "alpha": 1,
        "mu_scale": 1.8,
        "order": 3,
        "max_t": 60,
        "dt": 1e-3,
        "modes": 21,
        "timestepper": "rk4",
        "initial_guess": {"lambda2": 2}}
    ranges = {
        "alpha": [.1, .5, 1, 5, 10, 50, 100],
        "mu_scale": [5e-4, .001, .005, .01, .05, .1, .5, 1, 1.4, 1.8]
    }
    sim.run_batch(base_params, ranges=ranges, grid=False)  # No grid search.


@option
def multiparam_mu_alpha_convergence(dirname):
    """Vary mu_scale and alpha - and see how the convergence rate changes.
    """
    sim = BatchSimulator(dirname, N=512, lambda2=1, overwrite=True)
    base_params = {
        "alpha": 1,
        "mu_scale": 1.8,
        "order": 3,
        "max_t": 60,
        "dt": 1e-3,
        "modes": 21,
        "timestepper":"rk4",
        "initial_guess": {"lambda2": 2, "lambda4":2, "nonlinear_coeff":2}
    }
    ranges = {
        "alpha": [.1, .5, 1, 5, 10, 50, 100],
        "mu_scale": [5e-4, .001, .005, .01, .05, .1, .5, 1, 1.4, 1.8]
    }
    sim.run_batch(base_params, ranges=ranges, grid=False)


@option
def interpolator_scan(dirname):
    """Do a parameter scan over interpolators.

    We vary both the functional form (piecewise-linear, cubic, Fourier modes,
    etc) and the dimension.
    """

    sim = BatchSimulator(dirname, N=512, lambda2=1, overwrite=True)
    # modes = [4, 8, 16, 21, 32, 48, 64, 96, 128]
    # interpolators = [None, "cubic", "linear"]
    # None stands for the standard Fourier interpolator.
    # Use a much smaller mu.
    base_params = {
        "alpha": 1,
        "mu_scale": .01,
        "order": 3,
        "max_t": 60,
        "dt": 1e-3,
        "modes": 21,
        "timestepper":"rk4",
        "initial_guess": {"lambda2": 2}
    }
    pointwise_modes = list(range(34,48,2))
    fourier_modes = list(range(17,22))
    mu_scales = [1e-3, 1e-2, 1e-1]
    pointwise_params = sim.get_param_list(base_params, ranges={
        "modes": pointwise_modes,
        "pointwise_interpolation": ["cubic", "quadratic", "linear"],
        "mu_scale": [.01]}
    )
    fourier_params = sim.get_param_list(base_params, ranges={
        "modes":fourier_modes,
        "mu_scale": [1.8]}
    )
    # sim.run_batch(base_params, ranges={
    #     "modes": modes,
    #     "pointwise_interpolation": interpolators}
    # )
    sim.run_simulations_low(pointwise_params + fourier_params)


# =============================================================================
if __name__ == "__main__":
    func_names = list(func_dict.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("dirname",
                        help="Directory in which to save simulation results.")
    parser.add_argument("--simulation",
                        choices=func_names, default="probe_order")
    args = parser.parse_args()
    print(f"Running simulation {args.simulation}")
    func_dict[args.simulation](args.dirname)
