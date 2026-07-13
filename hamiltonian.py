"""
hamiltonian.py
==============

Defines Hamiltonian systems.

Every Hamiltonian system must inherit from the Hamiltonian base class
and implement

    H(q,p)
    dHdq(q,p)
    dHdp(q,p)

Integrators never need to know which Hamiltonian they are solving.
"""

from abc import ABC, abstractmethod
import numpy as np


# ============================================================
# Base Hamiltonian
# ============================================================

class Hamiltonian(ABC):
    """
    Abstract Hamiltonian system.

    Any Hamiltonian system should inherit from this class.
    """

    @abstractmethod
    def H(self, q, p):
        """
        Hamiltonian.
        """
        pass

    @abstractmethod
    def dHdq(self, q, p):
        """
        ∂H/∂q
        """
        pass

    @abstractmethod
    def dHdp(self, q, p):
        """
        ∂H/∂p
        """
        pass

    # --------------------------------------------------------

    def equations_of_motion(self, q, p):
        """
        Hamilton's equations.

        Returns
        -------
        dq/dt , dp/dt
        """

        dq = self.dHdp(q, p)
        dp = -self.dHdq(q, p)

        return dq, dp


# ============================================================
# Harmonic Oscillator
# ============================================================

class HarmonicOscillator(Hamiltonian):

    def __init__(self, m=1.0, k=1.0):

        self.m = m
        self.k = k

    def H(self, q, p):

        return p**2/(2*self.m) + 0.5*self.k*q**2

    def dHdq(self, q, p):

        return self.k*q

    def dHdp(self, q, p):

        return p/self.m


# ============================================================
# Pendulum
# ============================================================

class Pendulum(Hamiltonian):

    def __init__(self, m=1.0, l=1.0, g=1.0):

        self.m = m
        self.l = l
        self.g = g

    def H(self, q, p):

        kinetic = p**2/(2*self.m*self.l**2)

        potential = self.m*self.g*self.l*(1 - np.cos(q))

        return kinetic + potential

    def dHdq(self, q, p):

        return self.m*self.g*self.l*np.sin(q)

    def dHdp(self, q, p):

        return p/(self.m*self.l**2)


# ============================================================
# Quartic Oscillator
# ============================================================

class QuarticOscillator(Hamiltonian):

    """
    H = p²/2 + q⁴/4
    """

    def H(self, q, p):

        return 0.5*p**2 + 0.25*q**4

    def dHdq(self, q, p):

        return q**3

    def dHdp(self, q, p):

        return p


# ============================================================
# Double Well
# ============================================================

class DoubleWell(Hamiltonian):

    """
    H = p²/2 + q⁴/4 - q²/2
    """

    def H(self, q, p):

        return 0.5*p**2 + 0.25*q**4 - 0.5*q**2

    def dHdq(self, q, p):

        return q**3 - q

    def dHdp(self, q, p):

        return p

