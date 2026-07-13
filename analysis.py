"""
analysis.py
===========

Analysis routines for Hamiltonian trajectories.

These functions analyse trajectories after numerical integration.

Current diagnostics
-------------------
• Energy conservation
• Centroid of a cloud
• Covariance matrix
• RMS cloud radius
"""

import numpy as np


# ============================================================
# ENERGY
# ============================================================

def energy_error(traj):
    """
    Energy error relative to the initial energy.

    Returns
    -------
    ndarray
        Shape (Nt, Nparticles)
    """

    return traj.energy - traj.energy[0]


# ------------------------------------------------------------

def max_energy_error(traj):
    """
    Maximum absolute energy error.
    """

    return np.max(np.abs(energy_error(traj)))


# ------------------------------------------------------------

def rms_energy_error(traj):
    """
    RMS energy error over all particles and timesteps.
    """

    err = energy_error(traj)

    return np.sqrt(np.mean(err**2))


# ============================================================
# CLOUD GEOMETRY
# ============================================================

def centroid(q, p):
    """
    Centroid of a cloud.

    Returns
    -------
    (qc, pc)
    """

    return np.mean(q), np.mean(p)


# ------------------------------------------------------------

def centroid_history(traj):
    """
    Centroid history.

    Returns
    -------
    qc, pc
    """

    qc = np.zeros(traj.nsteps)
    pc = np.zeros(traj.nsteps)

    for i in range(traj.nsteps):

        qc[i], pc[i] = centroid(
            traj.q[i],
            traj.p[i]
        )

    return qc, pc


# ------------------------------------------------------------

def covariance_matrix(q, p):
    """
    Covariance matrix of one cloud.
    """

    return np.cov(np.vstack((q, p)))


# ------------------------------------------------------------

def covariance_history(traj):
    """
    Covariance matrix at every timestep.

    Returns
    -------
    ndarray
        Shape (Nt,2,2)
    """

    cov = np.zeros((traj.nsteps, 2, 2))

    for i in range(traj.nsteps):

        cov[i] = covariance_matrix(
            traj.q[i],
            traj.p[i]
        )

    return cov


# ------------------------------------------------------------

def rms_radius(q, p):
    """
    RMS distance from the centroid.
    """

    qc, pc = centroid(q, p)

    r2 = (q - qc)**2 + (p - pc)**2

    return np.sqrt(np.mean(r2))


# ------------------------------------------------------------

def rms_radius_history(traj):
    """
    RMS cloud radius as a function of time.
    """

    r = np.zeros(traj.nsteps)

    for i in range(traj.nsteps):

        r[i] = rms_radius(
            traj.q[i],
            traj.p[i]
        )

    return r


# ============================================================
# PLACEHOLDER FOR FUTURE AREA ROUTINES
# ============================================================

def phase_area(traj):
    """
    Placeholder.

    Later this will compute the area of the cloud
    using a robust algorithm (likely Delaunay based).

    Raises
    ------
    NotImplementedError
    """

    raise NotImplementedError(
        "Phase-space area estimation has not yet been implemented."
    )


# ============================================================
# SUMMARY
# ============================================================

def summary(traj):

    print()
    print("Trajectory summary")
    print("------------------")

    print(f"Steps          : {traj.nsteps}")
    print(f"Particles      : {traj.nparticles}")

    print(f"Maximum |ΔE|   : {max_energy_error(traj):.4e}")
    print(f"RMS ΔE         : {rms_energy_error(traj):.4e}")

    print()