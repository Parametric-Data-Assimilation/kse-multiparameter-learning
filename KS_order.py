# KS_order.py
"""
"""

import numpy as np
import numpy.linalg as la

from finite_difference import stable_fdcoeffs
import imex_timesteppers as imex
import matplotlib.pyplot as plt

class KS:
    """Solver for the 1-d Kuramoto-Sivashinsky equation, the simplest PDE that
    exhibits spatio-temporal chaos:

    u_t + nonlinear_coeff*u*u_x + lambda2*u_xx + lambda4*u_xxxx = 0,
    periodic BCs on [0,2πL].

    time step dt with N fourier collocation points.

    energy enters the system at long wavelengths via lambda2*u_xx,
    cascades to short wavelengths due to the nonlinearity u*u_x,
    and dissipates via lambda4*u_xxxx.
    The greater the lambda2 parameter, the more chaotic the system.

    This solver also allows the injection of additional terms u_x and u_xxx,
    with coefficients lambda1 and lambda3, respectively.

    https://www.encyclopediaofmath.org/index.php/Kuramoto-Sivashinsky_equation
    """
    def __init__(self, L=16, N=128, dt=0.5,
                 lambda2=1.0, lambda4=1, nonlinear_coeff=1,
                 lambda1=0, lambda3=0, nonlinear_coeff2=0,
                 nonlinear_coeff3=0,
                 timestepper='rk3'):
        """Initialize the solver.

        Parameters
        ----------

        L : float
            Length of the domain is 2πL.

        N : int
            Number of Fourier collocation points (resolution of the solution
            representation).

        dt : float
            Time step.

        lambda2 : float
            Coefficient on the u_xx term in the equation.

        lambda4 : float
            Coefficient on the u_xxxx term in the equation.
            One for the standard Kuramoto-Sivashinsky equation.

        nonlinear_coeff : float
            Coefficient on the u*u_x term in the equation.
            One for the standard Kuramoto-Sivashinsky equation.

        lambda1 : float
            Coefficient on an additional u_x term in the equation.
            Zero for the standard Kuramoto-Sivashinsky equation.

        lambda3 : float
            Coefficient on an additional u_xxx term in the equation.
            Zero for the standard Kuramoto-Sivashinsky equation.

        nonlinear_coeff2 : float
            Coefficient on the u^2 term in the equation.
            Zero for the standard Kuramoto-Sivashinsky equation.

        nonlinear_coeff3 : float
            Coefficient on the u_x^2 term in the equation.
            Zero for the standard Kuramoto-Sivashinsky equation.

        timestepper : str
            Which time stepping scheme to use. Options:
            * 'forward_euler': First-order Euler method.
            * 'rk3': Third-order Runge-Kutta method.
        """
        self.L, self.n, self.dt = L, N, dt
        self.lambda2, self.timestepper = lambda2, timestepper
        kk = N*np.fft.fftfreq(N)[0:int(N//2 + 1)]      # Wave numbers
        self.k = kk.astype(np.float) / L

        # Store coefficients and initialize linear Fourier multiplier.
        self.lambda4 = lambda4
        self.nonlinear_coeff = nonlinear_coeff
        self.nonlinear_coeff2 = nonlinear_coeff2
        self.nonlinear_coeff3 = nonlinear_coeff3
        self.lambda1 = lambda1
        self.lambda3 = lambda3
        self.update_lin()

        # Set the (fixed) initial condition.
        x = (np.pi/L)*np.linspace(-L, L , N+1)[:-1]
        x =       np.sin(6*x)  +  .1*np.cos(x)    - .2*np.sin(3*x) \
            + .05*np.cos(15*x) + 0.7*np.sin(18*x) -    np.cos(13*x)

        # Process initial condition.
        self.x = x - x.mean()                       # Remove zonal mean.
        self.xspec = np.fft.rfft(self.x)            # Spectral space variable.

        # initialize timestepper class if using RK4
        if timestepper == 'rk4':
            self.rk4 = imex.RK664()


    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, k):
        self.__k = k
        self.__ik = 1j*k
        self.__ik[-1] = 0

    @property
    def ik(self):
        """Spectral derivative operator."""
        return self.__ik

    def get_domain(self):
        """Get the domain for plotting the solution in real space.
        """

        return np.pi*np.linspace(-self.L, self.L , self.n+1)[:-1]

    def nlterm(self, xspec, return_specs=False):
        """Compute the tendency from the nonlinear term."""
        xspec_dealiased = xspec.copy()
        xspec_dealiased[(2*len(xspec))//3:] = 0
        x = np.fft.irfft(xspec_dealiased)
        res = -0.5*self.ik*np.fft.rfft(x**2) * self.nonlinear_coeff
        # We need to flip the sign for the specs
        # Since the parameter estimation needs a different sign than the timestepping
        specs = {}
        specs['nonlinear_coeff'] = -res.copy()
        if self.nonlinear_coeff2:
            specs['nonlinear_coeff2'] = self.nonlinear_coeff2 * np.fft.rfft(x*np.fft.irfft(self.ik**2*xspec_dealiased))
            res -= specs['nonlinear_coeff2']
        if self.nonlinear_coeff3:
            u_xspec = - self.ik * xspec_dealiased
            specs['nonlinear_coeff3'] = self.nonlinear_coeff3 * np.fft.rfft(np.fft.irfft(u_xspec)**2)
            res -= specs['nonlinear_coeff3']

        if return_specs:
            return res, specs
        else:
            return res

    def update_lin(self):
        """Set Fourier multipliers for the linear terms."""
        self.lin = (- self.lambda1*self.ik        # u_x
                    + self.lambda2*self.k**2              # u_xx
                    - self.lambda3 * self.ik**3   # u_xxx
                    - self.lambda4*self.k**4)                  # u_xxxx

    def advance(self):
        """Advance one time step by operating in Fourier space."""
        self.xspec = np.fft.rfft(self.x)                    # DFT
        self._do_time_step()                                # Step in Fourier
        self.x = np.fft.irfft(self.xspec)                   # Inverse DFT

    def _do_time_step(self):
        """Do a single integration step via the Fourier coefficients."""
        xspec_save = self.xspec.copy()
        if self.timestepper == 'rk3':
            # Semi-implicit third-order runge kutta update.
            # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
            for n in range(3):
                dt = self.dt/(3-n)
                # Explicit RK3 step for nonlinear term.
                self.xspec = xspec_save + dt*self.nlterm(self.xspec)
                # Implicit trapezoidal adjustment for linear term.
                self.xspec = (self.xspec+0.5*self.lin*dt*xspec_save)/(1.-0.5*self.lin*dt)
        elif self.timestepper == 'rk4':
            self.xspec = self.rk4.step(xspec_save, self.dt, self.nlterm, self.lin)
        elif self.timestepper == 'forward_euler':
            # Forward Euler (avoiding any issues with RK etc. methods).
            dt = self.dt
            self.xspec = xspec_save + dt*self.nlterm(xspec_save) + dt*self.lin   
        else:
            raise ValueError(f"Unrecognized timestepper {self.timestepper}!")


def fourier_inner_product(spec1, spec2):
    """Compute the REAL inner product of two vectors in Fourier space by first
    applying the inverse DFT, i.e., <iDFT(spec1), iDFT(spec2)>_{2}.
    """
    return np.fft.irfft(spec1) @ np.fft.irfft(spec2)
    # return np.real(spec1.conj() @ spec2) # This is *not* the same!


class KSAssim(KS):
    """Perform KSE data assimilation on the PDE level.

    Given spatially limited observations of a solution with unkown initial
    state and viscosity, we seek to recover the true viscosity and model state
    (marching forward in time).

    This requires solving a modified version of the original PDE that directly
    incorporates the observations from the true state.

    Additionally, the viscosity is initially unknown. We start with an initial
    guess and update it on the fly.
    """
    def __init__(self, projector, mu=1, alpha=None,
                 estimate_params=('lambda2',), order=1, **kwargs):
        """mu is the weight controlling the data assimilation
        alpha, if provided will nudge the parameters towards the estimated parameter, instead of jumping at each timestep
        This way the inital guess has some effect.
        The order is the order of finite difference to use in approximating u_t

        Parameters
        ---------
        projector : TODO

        mu : float
            Weight of the data assimilation (relaxation hyperparameter).

        alpha : float or None
            If provided, nudge the parameters towards the estimated parameter.
            If None, jump at each timestep.

        estimate_params : list(str)
            List of parameters to estimate. Valid entries:
            * "lambda2"
            * TODO

        order : int
            Finite difference order used to approximate time derivative u_t.

        **kwargs : dict
            Arguments to initialize the solver. See KS.__init__().
        """
        KS.__init__(self, **kwargs)
        self.mu = mu
        self.projector = projector
        self.target_history = [None] * (order+1)
        self.target_spec = None
        self.order = order
        self.finite_difference_coeffs = stable_fdcoeffs(0, np.arange(-order, 1, 1), 1)
        # The ordering of the estimate params matters because it determines how the basis vectors are selected.
        # When solving the linear system (for multi-parameter estimation).
        # This ordering (and choice of basis) can affect the convergence of the overall algorithm.
        # If the nonlinear term is used as a basis function - it can break convergence
        # So we choose the ordering that works better
        self.estimate_params = self.sort_estimate_params(estimate_params)
        self.alpha = alpha
        self.param_coeffs = {"lambda4": self.k**4,
                             "lambda2": -self.k**2,
                             "lambda1": self.ik,
                             "lambda3": self.ik**3}

    def update_params(self, params):
        """Update the system parameters mid-simulation.

        Parameters
        ----------
        params : dict
            Dictionary mapping attribute name to new values. Valid keys:
            * lambda1
            * lambda2
            * lambda3
            * lambda4
            * nonlinear_coeff
        """
        for p in params:
            new_val = params[p]
            old_val = getattr(self, p)
            if self.alpha is not None:
                # use higher order estimate, i.e., trapezoid rule
                new_val = ((1-self.alpha*self.dt/2)*old_val + self.alpha*self.dt*new_val)/(1+self.alpha*self.dt/2)
    #                    new_val = old_val + self.dt*self.alpha*(new_val-old_val)
            setattr(self, p, new_val)
    #            print("Updated {} to {}".format(p, new_val))
        self.update_lin()

    def interpolate(self, spec):
        return np.fft.rfft(self.projector(spec))

    def advance(self):
        self.xspec = np.fft.rfft(self.x)

        w = (self.target_spec - self.interpolate(self.xspec))
        n_est_params = len(self.estimate_params)
        if self.target_history[0] is not None and n_est_params:
            # Need to compute time derivative of the projections of the true model state.
    #            u_t = (self.target_spec-self.last_target_spec) / self.dt
            # higher order 1-sided approximate of the derivative
            #u_t = .5*(3.*self.target_spec - 4.*self.last_target_spec + self.backstep_target_spec) / self.dt
            u_t = sum([coeff*spec for coeff, spec in zip(self.finite_difference_coeffs, self.target_history)]) / self.dt
    #            u_t = (11*self.target_spec/6 - 3*self.last_target_spec + 1.5*self.backstep_target_spec - self.back2step_target_spec/3)/self.dt
            _ , nl_specs = self.nlterm(self.xspec, return_specs=True)
            G_specs = {p: self.param_coeffs[p]*self.xspec for p in self.param_coeffs}
            G_specs.update(nl_specs)
            rhs = np.zeros_like(nl_specs['nonlinear_coeff'])

            for p in G_specs:
                if p not in self.estimate_params:
                    rhs += getattr(self, p)*G_specs[p]
            rhs_contrib = self.interpolate(rhs)
            # If we have one parameter, there is no need to call a linear system solver.
            if n_est_params == 1:
                    param = self.estimate_params[0]
                    G = self.interpolate(G_specs[param]) #Contribution from the linear diffusive term
                    num = fourier_inner_product(-w,u_t+rhs_contrib)
                    denom = fourier_inner_product(w, G)
                    #print(num/denom)
                    self.update_params({param:num/denom})
            else:
                #TODO: figure out how to better pick the other vectors
                for p in self.param_coeffs:
                    G_specs[p] = self.interpolate(G_specs[p])

                e_list = [w.copy()] + [G_specs[p].copy() for p in self.estimate_params]
                self.gram_schmidt(e_list)
                #print(e_list)
                #print(xspec_save, w)
                A = np.zeros((n_est_params,n_est_params))
                b = np.zeros(n_est_params)
                for i in range(n_est_params):
                    for k in range(n_est_params):
                        p = self.estimate_params[k]
                        A[i,k] = fourier_inner_product(G_specs[p], e_list[i])
                    b[i] = fourier_inner_product(-e_list[i], u_t+rhs_contrib)
                #print(A, la.det(A))
                estimate = la.solve(A,b)
                #print(estimate)
                self.update_params({p:estimate[i] for i,p in enumerate(self.estimate_params)})

        self._do_time_step()

        # forward Euler rather than RK3 to hopefully keep everything consistent
        #self.xspec += self.dt * (self.mu * w + self.nlterm(self.xspec) + self.lin)
        self.xspec += self.dt * self.mu * w
        self.x = np.fft.irfft(self.xspec)

    def set_target(self, target_projection):
        for i in range(len(self.target_history)-1):
            self.target_history[i] = self.target_history[i+1]
        self.target_spec = self.target_history[-1] = np.fft.rfft(target_projection)

    def gram_schmidt(self, vecs):
        """Modified Gram-Schmidt orthonormalization *IN-PLACE*.

        Parameters
        ----------
        vecs : list of ndarrays
            Vectors to orthonormalize.
        """
        for i in range(len(vecs)):
            # Normalize current vector
            vecs[i] /= fourier_inner_product(vecs[i], vecs[i])**.5
            for j in range(i+1, len(vecs)):
                # Orthogonalize remaining vectors
                vecs[j] -= fourier_inner_product(vecs[j], vecs[i]) * vecs[i]

    def error(self, true, kind='l2'):
        """Compute the l2 error of the truth and the current estimate in
        Fourier space.
        """
        errs = true.xspec - self.xspec
        return np.sqrt(np.sum(np.abs(errs)**2))

    def sort_estimate_params(self, params):
        # Ensure the nonlinear coefficient is always last
        return sorted(params, key=lambda p: 1 if 'nonlinear' in p else 0)

    def plot_spectrum(self):
        plt.semilogy(np.abs(self.k), np.abs(self.xspec))
        plt.title("KSE spectrum")
        plt.show()

# =============================================================================
def _test_KS():
    """Test the KSE solver with default parameters."""

    import tqdm

    dt = .0001
    N = 256
    kse = KS(dt=dt, N=N, lambda2=10)

    # Advance the solution far enough to see some chaos.
    max_n = int(2/dt)
    print(f"Time stepping solution {max_n} times")
    snapshots = np.empty((max_n, N))
    for n in tqdm.tqdm(range(max_n)):
        snapshots[n] = kse.x.copy()
        kse.advance()

    # Plot the solution in space-time.
    fig1, ax1 = plt.subplots(1, 1, figsize=(6,6))
    ax1.pcolormesh(snapshots)
    ax1.set_xlabel(r"Space $x$ (node points)")
    ax1.set_ylabel(r"Time $t$ (time steps)")
    fig1.tight_layout()

    # Plot the final solution in space.
    fig2, ax2 = plt.subplots(1, 1, figsize=(8,3))
    ax2.plot(kse.x, label='true')
    ax2.set_xlabel(r"Space $x$ (node points)")
    ax2.set_ylabel(r"Solution $u(x)$")
    fig2.tight_layout()

    plt.show()


# TODO:
#   - Merge KS class with other files; ensure correct class hierarchy
#   - Rename x to u, create spatial variable
#   - Add built-in visualization routines
#   - Rename system coefficients to lambda1, lambda2, etc.
#   - Documentation and example notebook
