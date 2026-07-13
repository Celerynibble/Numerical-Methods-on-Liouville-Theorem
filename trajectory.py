
"""
trajectory.py
-------------

Stores the complete evolution of a Hamiltonian system.

The trajectory may correspond to

    • a single particle
    • a cloud of particles

Internally,

    q.shape = (Nt, Nparticles)

If Nparticles = 1 this still works naturally.
"""

import numpy as np


class Trajectory:

    def __init__(self):

        # Time
        self.t = []

        # Phase-space coordinates
        self.q = []
        self.p = []

        # Hamiltonian evaluated at every particle
        self.energy = []

        # ---------- Future diagnostics ----------
        # These remain empty until computed.

        self.phase_area = None
        self.jacobian_det = None
        self.symplectic_error = None

    # ---------------------------------------------------------

    def append(self, t, q, p, H):

        """
        Append one timestep.

        Parameters
        ----------
        t : float

        q : ndarray
            Shape (N,)

        p : ndarray
            Shape (N,)

        H : callable
            Hamiltonian function.
        """

        q = np.asarray(q).copy()
        p = np.asarray(p).copy()

        self.t.append(t)
        self.q.append(q)
        self.p.append(p)

        self.energy.append(H(q, p).copy())

    # ---------------------------------------------------------

    def finalize(self):

        """
        Convert internal Python lists into NumPy arrays.
        """

        self.t = np.asarray(self.t)

        self.q = np.asarray(self.q)
        self.p = np.asarray(self.p)

        self.energy = np.asarray(self.energy)

    # ---------------------------------------------------------

    @property
    def nsteps(self):

        return len(self.t)

    @property
    def nparticles(self):

        if len(self.q) == 0:
            return 0

        return self.q.shape[1]

    # ---------------------------------------------------------

    @property
    def final_q(self):

        return self.q[-1]

    @property
    def final_p(self):

        return self.p[-1]

    @property
    def final_energy(self):

        return self.energy[-1]

    # ---------------------------------------------------------

    def __repr__(self):

        return (
            f"Trajectory("
            f"steps={self.nsteps}, "
            f"particles={self.nparticles})"
        )