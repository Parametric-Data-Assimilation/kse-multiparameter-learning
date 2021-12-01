# simulation_results.py
"""Classes for loading / visualizing data from simulations."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Result:
    """Single simulation result."""
    def __init__(self, params, arrays):
        self.params = params
        self.arrays = arrays

        # Verify arrays are consistent lengths.
        expected_len = None
        for k in self.arrays:
            ln = len(self.arrays[k])
            if expected_len is None:
                expected_len = ln
                continue
            if ln != expected_len:
                raise RuntimeWarning(f"Inconsistent length {ln} for"
                                     f"metric {k}! Expected {expected_len}")

        self.num_timesteps = expected_len

    def final_errors(self):
        """Calculate the mean error over the final second."""
        res = {}
        for k in self.arrays:
            # return average error over TODO
            res[k] = np.mean(self.arrays[k][-int(1./self.params["dt"]):])
        return res

    def determine_convergence_rate(self):
        """Estimate the convergence rate."""
        true_errs = self.arrays["true_errors"]
        steps_per_second = int(1./self.params["dt"])
        final_err = self.final_errors()["true_errors"]
        convergence_mask = true_errs < final_err
        try:
            steps_to_convergence = np.where(convergence_mask)[0][0]
        except IndexError:
            return np.nan
        seconds_to_convergence = steps_to_convergence/steps_per_second
        initial_err = true_errs[0]
        exp_decay_rate = (np.log(final_err) - np.log(initial_err))
        exp_decay_rate /= seconds_to_convergence
        return -exp_decay_rate

    def plot(self, ax=None, save_file=None, params_only=False, **kwargs):
        """Plot the result against time.

        Parameters
        ----------
        ax : plt.Axes or None
            Axes handle on which to plot. Figure created if not provided.

        Returns
        -------
        ax : plt.Axes
            Axes handle for the active subplot.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, **kwargs)

        domain = self.params['dt'] * np.arange(self.num_timesteps)
        for k in self.arrays:
            if params_only and "lambda" not in k:
                continue
            ax.semilogy(domain, self.arrays[k], lw=1, label=k)

        ax.set_xlim(domain[0], domain[-1])
        return ax


class SimulationResults():
    """Group multiple simulation results."""

    def __init__(self, dirname):
        with open(dirname+"/index.json", "r") as fp:
            index = json.load(fp)

        self.failed_simulations = []
        self.results = []
        for rec in index:
            if 'error' in rec:
                self.failed_simulations.append(rec)
                continue
            fname = dirname+"/"+rec['filename'].split("/")[-1]
            ark = np.load(fname)
            self.results.append(Result(rec["params"], ark))

    def __getitem__(self, ind):
        return self.results[ind]

    def plot_all(self):
        """Plot each of the result sets in different figures."""
        for r in self.results:
            print(r.params)
            r.plot()

    def get_summary(self):
        arr = []
        for r in self.results:
            errs = r.final_errors()
            rec = {k: v for k, v in r.params.items() if type(v) != dict}
            for k, v in r.params["initial_guess"].items():
                rec[k+"_initial_guess"] = v
            for k, v in errs.items():
                rec[k+"_error" if "error" not in k else k.replace("errors", "error")] = v

            rec["convergence_rate"] = r.determine_convergence_rate()
            arr.append(rec)
        return pd.DataFrame(arr)

    def get_result(self, search_params):
        for r in self.results:
            if all(r.params[k] == v for k,v in search_params.items()):
                print("Found matching result", r.params)
                return r


if __name__ == "__main__":
    # Test
    s = SimulationResults("data/multiparam_diffusion1")
