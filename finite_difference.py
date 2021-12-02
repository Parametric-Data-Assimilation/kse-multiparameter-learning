# finite_difference.py
"""This file contains a utility function, stable_fdcoeffs(), which generates
coefficients for approximating time derivatives via finite difference.
The function was translated from MATLAB and is numerically stable.
Based on [TODO]
"""

import numpy as np


def stable_fdcoeffs(xbar, x, k):
    """Generate coefficients to approximate the k-th derivative of a function
    f given its values at a non-uniform set of points.

    Parameters
    ----------
    xbar : float
        Point at which to approximate the derivative.
    x : (n,) ndarray
        Points at which the value of the function is known.
    k : int
        Order of the derivative to approximate.

    Returns
    -------
    C : (n,) ndarray
        Coefficients for the kth order finite difference approximation of the
        kth derivative of f at xbar. That is, f^(k)(xbar) = C @ f(x).
    """
    n = len(x)
    if k >= n:
        raise ValueError("len(x) must be larger than k")

    # change to m = n-1 if you want to compute coefficients for all
    # possible derivatives. Then modify to output all of C.
    m = k

    # Add padding so the 1-based indexing actually works
    x_new = np.zeros(len(x)+1)
    x_new[1:] = x
    x = x_new
    c1 = 1
    c4 = x[1] - xbar
    # Add an extra row and column so that the 1-based indexing doesn't break
    C = np.zeros((n-1+1+1,m+1+1))
    C[1,1] = 1
    for i in range(1,n):
        i1 = i+1
        mn = min(i,m)
        c2 = 1
        c5 = c4
        c4 = x[i1] - xbar
        for j in range(0,i):
            j1 = j+1
            c3 = x[i1] - x[j1]
            c2 = c2*c3
            if j == i-1:
                for s in list(range(1, mn+1))[::-1]:
                    s1 = s+1
                    C[i1,s1] = c1*(s*C[i1-1,s1-1] - c5*C[i1-1,s1])/c2
                C[i1,1] = -c1*c5*C[i1-1,1]/c2
            for s in list(range(1, mn+1))[::-1]:
                s1 = s+1
                C[j1,s1] = (c4*C[j1,s1] - s*C[j1,s1-1])/c3
            C[j1,1] = c4*C[j1,1]/c3
        c1 = c2
    return C[1:,-1]


def _test_stable_fdcoeffs(ntests=10):
    """Test stable_fdcoeffs() on a simple problem."""
    # Function to test.
    def f(x):
        return np.exp(x)*np.sin(x)

    def df(x):
        return np.exp(x)*(np.sin(x) + np.cos(x))

    def ddf(x):
        return 2*np.exp(x)*np.cos(x)

    # Do several random tests.
    for _ in range(ntests):

        # Random non-uniform test gridpoints.
        x = np.sort(np.random.standard_normal(10))
        fx = f(x)

        # Test first-order derivative.
        C = stable_fdcoeffs(x[1], x, 1)
        assert abs(fx @ C - df(x[1])) < 1e-3

        # Test second-order derivative.
        C = stable_fdcoeffs(x[2], x, 2)
        assert abs(fx @ C - ddf(x[2])) < 1e-3
