# Data Assimilation and Parameter Estimation for the Kuramoto-Sivashinsky Equation

This repository is the source code for the paper [_Concurrent MultiParameter Learning Demonstrated on the Kuramoto-Sivashinsky Equation_](https://epubs.siam.org/doi/abs/10.1137/21M1426109) by [Pachev](https://scholar.google.com/citations?user=QSTIQA4AAAAJ), [Whitehead](https://scholar.google.com/citations?user=lLR_YEYAAAAJ), and [McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ).


## Contents

- [KS_order.py](./KS_order.py): Main source file for current simulations.
- [batch_simulations.py](./batch_simulations.py): Run several simulation and save output data in `.npz` files.
- [paper_simulations.py](./paper_simulations.py): Simulations chosen for the paper.
- [finite_difference.py](./finite_difference.py): Defines a function `stable_fdcoeffs()` for computing the finite difference coefficients to estimate a function derivative at a point given function values on a nonuniform grid. Used by `KS_order.KSAssim`.
- [simulation_results.py](./simulation_results.py): Defines a convenience class for reading / visualizing data saved by `batch_simulations.py`.
<!-- - [run_background_simulation.sh](./run_background_simulation.sh): -->


## Citation

If you find this repository useful, please consider citing our paper:

[Pachev, B.](https://scholar.google.com/citations?user=QSTIQA4AAAAJ), [Whitehead, J.](https://scholar.google.com/citations?user=lLR_YEYAAAAJ), and [McQuarrie, S. A.](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [**Concurrent multiparameter learning demonstrated on the Kuramoto-Sivashinsky equation**](https://epubs.siam.org/doi/abs/10.1137/21M1426109). _SIAM Journal on Scientific Computing_, 44(5):A2974–A2990, 2022.
```
@article{pachev2022multiparameter,
  author = {Pachev, Benjamin and Whitehead, Jared P. and McQuarrie, Shane A.},
  title = {Concurrent MultiParameter Learning Demonstrated on the {K}uramoto--{S}ivashinsky Equation},
  journal = {SIAM Journal on Scientific Computing},
  volume = {44},
  number = {5},
  pages = {A2974-A2990},
  year = {2022},
  doi = {10.1137/21M1426109},
}
```
