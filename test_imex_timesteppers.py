from imex_timesteppers import *
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

def test_rk222():
    f = lambda u: np.array([u[1]**2, 0])
    g = np.array([0, 2])
    y0 = np.array([1./4, 1.])
    sol = solve_ivp(lambda t,y: f(y) + g * y, [0,2], [1./4, 1.],
        t_eval = np.linspace(0,2,201))

    plt.plot(sol.t, sol.y.T)
    plt.show()
    print(sol.t[1])
    print(sol.y[:, -1])
    h = 1e-2
    n = int(2/h)
    sol = np.zeros((len(y0), n+1))
    rk = RK222()
    y = y0
    for i in range(n):
        y = rk.step(y, h, f, g)
        sol[:, i+1] = y

    print(sol[:,-1])
    plt.plot(np.linspace(0,2, n+1), sol.T)
    plt.show()

def test_rk443():
    f = lambda u: np.array([u[1]**2, 0])
    g = np.array([0, 2])
    y0 = np.array([1./4, 1.])
    b = 1
    for h in [5e-2, 2.5e-2, 1.25e-2]:
        n = int(b/h)
        sol = np.zeros((len(y0), n+1))
        rk = RK664()
        y = y0
        sol[:,0] = y
        for i in range(n):
            y = rk.step(y, h, f, g)
            sol[:, i+1] = y

        x = np.linspace(0, b, n + 1)
        true_soln = np.array([.25*np.exp(4*x), np.exp(2*x)])
        print(np.max(np.abs(sol-true_soln)))
    #print(sol)
    #print(true_soln)

if __name__ == "__main__":
    test_rk443()
