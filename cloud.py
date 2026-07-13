"""
cloud.py
========

Represents a finite patch of phase space.

A Cloud is a collection of sample points representing a small
region of phase space. Numerical integrators evolve the sample
points, while analysis routines reconstruct the underlying region.
"""

import numpy as np


class Cloud:

    def __init__(self, q0, p0):

        q0 = np.asarray(q0, dtype=float)
        p0 = np.asarray(p0, dtype=float)

        if q0.shape != p0.shape:
            raise ValueError(
                "q0 and p0 must have identical shapes."
            )

        self.q0 = q0
        self.p0 = p0

    # =========================================================
    # Properties
    # =========================================================

    @property
    def nparticles(self):
        return len(self.q0)

    @property
    def points(self):
        """
        Return cloud as an (N,2) array.
        """

        return np.column_stack((self.q0, self.p0))

    # =========================================================
    # Constructors
    # =========================================================

    @classmethod
    def single_particle(cls, q, p):

        return cls([q], [p])

    # ---------------------------------------------------------

    @classmethod
    def disk(cls,
             center=(0.0,0.0),
             radius=0.05,
             N=500):
        """
        Uniform sampling of a disk.
        """

        q0,p0 = center

        theta = np.random.uniform(
            0,
            2*np.pi,
            N
        )

        r = radius*np.sqrt(
            np.random.rand(N)
        )

        q = q0 + r*np.cos(theta)
        p = p0 + r*np.sin(theta)

        return cls(q,p)

    # ---------------------------------------------------------

    @classmethod
    def circle(cls,
               center=(0.0,0.0),
               radius=0.05,
               N=300):
        """
        Uniform sampling of a circle boundary.
        """

        q0,p0 = center

        theta = np.linspace(
            0,
            2*np.pi,
            N,
            endpoint=False
        )

        q = q0 + radius*np.cos(theta)
        p = p0 + radius*np.sin(theta)

        return cls(q,p)

    # ---------------------------------------------------------

    @classmethod
    def gaussian(cls,
                 center=(0.0,0.0),
                 sigma=0.05,
                 N=500):

        q0,p0 = center

        q = np.random.normal(
            q0,
            sigma,
            N
        )

        p = np.random.normal(
            p0,
            sigma,
            N
        )

        return cls(q,p)

    # ---------------------------------------------------------

    @classmethod
    def rectangle(cls,
                  qmin,
                  qmax,
                  pmin,
                  pmax,
                  N=500):

        q = np.random.uniform(
            qmin,
            qmax,
            N
        )

        p = np.random.uniform(
            pmin,
            pmax,
            N
        )

        return cls(q,p)

    # ---------------------------------------------------------

    @classmethod
    def line(cls,
             q1,
             p1,
             q2,
             p2,
             N=500):

        q = np.linspace(q1,q2,N)
        p = np.linspace(p1,p2,N)

        return cls(q,p)

    # =========================================================
    # Utilities
    # =========================================================

    def copy(self):

        return Cloud(
            self.q0.copy(),
            self.p0.copy()
        )

    # ---------------------------------------------------------

    def __len__(self):

        return self.nparticles

    # ---------------------------------------------------------

    def __repr__(self):

        return (
            f"Cloud("
            f"{self.nparticles} sample points)"
        )