# test_imex_timesteppers.py
"""Tests for imex_timesteppers.py."""

import numpy as np
import scipy.integrate as sin
from matplotlib import pyplot as plt

import imex_timesteppers as imex


def test_rk222():
    """Test imex_timesteppers.RK222()."""

    def f(u):
        return np.array([u[1]**2, 0])

    g = np.array([0, 2])
    y0 = np.array([1./4, 1.])
    sol = sin.solve_ivp(lambda t,y: f(y) + g * y,
                        [0,2],
                        [1./4, 1.],
                        t_eval=np.linspace(0, 2, 201))

    plt.plot(sol.t, sol.y.T)
    plt.show()

    print(sol.t[1])
    print(sol.y[:, -1])
    h = 1e-2
    n = int(2/h)
    sol = np.zeros((len(y0), n+1))
    rk = imex.RK222()
    y = y0
    for i in range(n):
        y = rk.step(y, h, f, g)
        sol[:, i+1] = y

    print(sol[:,-1])
    plt.plot(np.linspace(0,2, n+1), sol.T)
    plt.show()


def test_rk664():
    """Test imex_timesteppers.RK664()."""

    def f(u):
        return np.array([u[1]**2, 0])

    g = np.array([0, 2])
    y0 = np.array([1./4, 1.])
    b = 1
    for h in [5e-2, 2.5e-2, 1.25e-2]:
        n = int(b/h)
        sol = np.zeros((len(y0), n+1))
        rk = imex.RK664()
        y = y0
        sol[:,0] = y
        for i in range(n):
            y = rk.step(y, h, f, g)
            sol[:, i+1] = y

        x = np.linspace(0, b, n + 1)
        true_soln = np.array([.25*np.exp(4*x), np.exp(2*x)])
        print(np.max(np.abs(sol-true_soln)))


if __name__ == "__main__":
    test_rk664()
