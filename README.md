# Kuramoto-Sivashinsky Equation in Python: Data Assimilation + Parameter Estimation

### Contents

- [KS_order.py](./KS_order.py): Main source file for current simulations.
- [batch_simulations.py](./batch_simulations.py): Run several simulation and save output data in `.npz` files.
- [paper_simulations.py](./paper_simulations.py): Simulations chosen for the paper.
- [finite_difference.py](./finite_difference.py): Defines a function `stable_fdcoeffs()` for computing the finite difference coefficients to estimate a function derivative at a point given function values on a nonuniform grid. Used by `KS_order.KSAssim`.
- [simulation_results.py](./simulation_results.py): Defines a convenience class for reading / visualizing data saved by `batch_simulations.py`.
<!-- - [run_background_simulation.sh](./run_background_simulation.sh): -->
