# imex_timesteppers.py
"""Implicit-explicit Runge-Kutta time-stepping methods, based off of the
dedalus package (https://dedalus-project.org/).
"""

import numpy as np


class RKIMEX:
    """Base class for IMEX Runge-Kutta time-stepping (a very simplified
    version of what dedalus uses). Applies to equations of the form

        U_t = G(u) + F(u)

    where F is nonlinear and does not depend on time.
    G is assumed to be linear and diagonal.
    """
    # Butcher tableau for the explicit scheme.
    A = NotImplemented
    # Butcher tableau for the implicit scheme.
    H = NotImplemented
    # Whether or not we need to take a linear combination of the stages.
    have_b = False

    def __init__(self):
        """Set stages."""
        self.stages = len(self.A)-1

    def step(self, u, h, f, g):
        """Compute one step of the IMEX Runge-Kutta method

        Parameters
        ----------
        u : ndarray
            Current state
        h : float
            Step size
        f : func
            Nonlinear term
        g : ndarray
            Representation of the linear term (assumed diagonal).
        """
        # Using the dedalus notation
        # nonlinear term applied to intermediate stages
        F = []
        # linear term applied to intermediate stages
        Lx = []
        u_curr = u.copy()
        for i in range(1, self.stages + 1):
            Lx.append(g*u_curr)
            F.append(f(u_curr))
            X = u.copy()
            for j in range(i):
                X += h * (self.A[i,j] * F[j] + self.H[i,j] * Lx[j])
            u_curr = X / (1 - h * self.H[i,i] * g)

        if not self.have_b:
            return u_curr
        else:
            Lx.append(g*u_curr)
            F.append(f(u_curr))
            X = u.copy()
            for i in range(len(F)):
                X += h*(self.b[i]*F[i] + self.b[i]*Lx[i])
            return X


class RK222(RKIMEX):
    γ = (2 - np.sqrt(2)) / 2
    δ = 1 - 1 / γ / 2

    A = np.array([[0,   0, 0],
                  [γ,   0, 0],
                  [δ, 1-δ, 0]])

    H = np.array([[0,   0, 0],
                  [0,   γ, 0],
                  [0, 1-γ, γ]])


class RK443(RKIMEX):
    A = np.array([[    0,    0,   0,    0, 0],
                  [  1/2,    0,   0,    0, 0],
                  [11/18, 1/18,   0,    0, 0],
                  [  5/6, -5/6, 1/2,    0, 0],
                  [  1/4,  7/4, 3/4, -7/4, 0]])

    H = np.array([[0,    0,    0,   0,   0],
                  [0,  1/2,    0,   0,   0],
                  [0,  1/6,  1/2,   0,   0],
                  [0, -1/2,  1/2, 1/2,   0],
                  [0,  3/2, -3/2, 1/2, 1/2]])


class RK664(RKIMEX):
    """The source for these horrendous coefficients is:
    Christopher A. Kennedy, Mark H. Carpenter
    Additive Runge–Kutta schemes for convection–diffusion–reaction equations
    (2003)

    See also https://scicomp.stackexchange.com/questions/21866/fourth-order-imex-runge-kutta-method
    """
    A = np.array([[0, 0, 0, 0, 0, 0],
                  [1/2, 0, 0, 0, 0, 0],
                  [13861/62500, 6889/62500, 0, 0, 0, 0],
                  [-116923316275/2393684061468,
                   -2731218467317/15368042101831,
                   9408046702089/11113171139209, 0, 0, 0],
                  [-451086348788/2902428689909,
                   -2682348792572/7519795681897,
                   12662868775082/11960479115383,
                   3355817975965/11060851509271, 0, 0],
                  [647845179188/3216320057751,
                   73281519250/8382639484533,
                   552539513391/3454668386233,
                   3354512671639/8306763924573, 4040/17871, 0]])

    H = np.array([[0, 0, 0, 0, 0, 0],
                  [1/4, 1/4, 0, 0, 0, 0],
                  [8611/62500, -1743/31250, 1/4, 0, 0, 0],
                  [5012029/34652500,
                   -654441/2922500,
                   174375/388108, 1/4, 0, 0],
                  [15267082809/155376265600,
                   -71443401/120774400,
                   730878875/902184768,
                   2285395/8070912, 1/4, 0],
                  [82889/524892,
                   0,
                   15625/83664,
                   69875/102672,
                   -2260/8211, 1/4]])

    b = H[-1]
    b_hat = np.array([4586570599/29645900160,
                      0,
                      178811875/945068544,
                      814220225/1159782912,
                      -3700637/11593932,
                      61727/225920])
    have_b = True
