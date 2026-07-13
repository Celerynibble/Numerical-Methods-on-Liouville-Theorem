"""
integrators.py
==============

Numerical integrators for Hamiltonian systems.

Every integrator inherits from Integrator and implements

    step(q,p)

The Integrator class provides the generic evolve() routine.

All integrators work with

    • single particles
    • clouds of particles

without modification.
"""

import numpy as np

from trajectory import Trajectory


# ============================================================
# Base Integrator
# ============================================================

class Integrator:

    def __init__(self, system, dt):

        self.system = system
        self.dt = dt

    # --------------------------------------------------------

    def step(self, q, p):

        raise NotImplementedError

    # --------------------------------------------------------

    def evolve(self, cloud, T):

        """
        Evolve a Cloud object.

        Parameters
        ----------
        cloud : Cloud

        T : float
            Final integration time.
        """

        q = cloud.q0.copy()
        p = cloud.p0.copy()

        nsteps = int(T / self.dt)

        traj = Trajectory()

        t = 0.0
        traj.append(t, q, p, self.system.H)

        for _ in range(nsteps):

            q, p = self.step(q, p)

            t += self.dt

            traj.append(t, q, p, self.system.H)

        traj.finalize()

        return traj


# ============================================================
# Euler
# ============================================================

class Euler(Integrator):

    def step(self, q, p):

        dq, dp = self.system.equations_of_motion(q, p)

        q_new = q + self.dt*dq
        p_new = p + self.dt*dp

        return q_new, p_new


# ============================================================
# RK4
# ============================================================

class RK4(Integrator):

    def F(self, q, p):

        dq, dp = self.system.equations_of_motion(q, p)

        return np.array([dq, dp])

    def step(self, q, p):

        y = np.array([q, p])

        k1 = self.F(*y)

        k2 = self.F(*(y + 0.5*self.dt*k1))

        k3 = self.F(*(y + 0.5*self.dt*k2))

        k4 = self.F(*(y + self.dt*k3))

        y = y + self.dt*(k1 + 2*k2 + 2*k3 + k4)/6

        return y[0], y[1]


# ============================================================
# Störmer-Verlet
# ============================================================

class StormerVerlet(Integrator):

    """
    Symplectic Störmer-Verlet integrator.

    Valid for separable Hamiltonians

        H(q,p)=T(p)+V(q)

    which includes

        • Harmonic oscillator
        • Pendulum
        • Quartic oscillator
        • Double well
        • Kepler
    """

    def step(self, q, p):

        p_half = p - 0.5*self.dt*self.system.dHdq(q, p)

        q_new = q + self.dt*self.system.dHdp(q, p_half)

        p_new = p_half - 0.5*self.dt*self.system.dHdq(q_new, p_half)

        return q_new, p_new
