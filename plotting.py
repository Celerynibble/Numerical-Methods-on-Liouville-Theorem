"""
plotting.py
===========

Plotting utilities for HamiltonianFlow.

This module only visualizes data.

All numerical analysis is performed in analysis.py.
"""

import matplotlib.pyplot as plt

import analysis


# ============================================================
# SINGLE PARTICLE PHASE SPACE
# ============================================================

def phase_space(
    traj,
    particle=0,
    ax=None,
    title=None,
    color="C0",
    lw=1.5
):
    """
    Plot one particle trajectory in phase space.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        traj.q[:, particle],
        traj.p[:, particle],
        color=color,
        lw=lw
    )

    ax.set_xlabel("q")
    ax.set_ylabel("p")

    ax.grid(True)
    ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)

    return ax


# ============================================================
# CLOUD
# ============================================================

def cloud(
    traj,
    step=-1,
    ax=None,
    title=None,
    color="C0",
    size=3,
    alpha=0.6
):
    """
    Scatter plot of a cloud at one timestep.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        traj.q[step],
        traj.p[step],
        s=size,
        c=color,
        alpha=alpha
    )

    ax.set_xlabel("q")
    ax.set_ylabel("p")

    ax.grid(True)
    ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)

    return ax


# ============================================================
# MULTIPLE CLOUD SNAPSHOTS
# ============================================================

def cloud_snapshots(
    traj,
    steps,
    labels=None,
    colors=None,
    ax=None,
    title=None,
    size=3,
    alpha=0.5
):
    """
    Plot several cloud snapshots on one axis.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green"]

    if labels is None:
        labels = [f"Step {i}" for i in steps]

    for step, color, label in zip(steps, colors, labels):

        ax.scatter(
            traj.q[step],
            traj.p[step],
            s=size,
            color=color,
            alpha=alpha,
            label=label
        )

    ax.set_xlabel("q")
    ax.set_ylabel("p")

    ax.grid(True)
    ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)

    ax.legend()

    return ax


# ============================================================
# ENERGY
# ============================================================

def energy(
    traj,
    particle=0,
    ax=None,
    title="Energy Error"
):
    """
    Plot energy error.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    err = analysis.energy_error(traj)

    ax.plot(
        traj.t,
        err[:, particle]
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$H-H_0$")

    ax.grid(True)

    ax.set_title(title)

    return ax


# ============================================================
# RMS CLOUD RADIUS
# ============================================================

def rms_radius(
    traj,
    ax=None,
    title="RMS Cloud Radius"
):
    """
    Plot RMS cloud radius.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    radius = analysis.rms_radius_history(traj)

    ax.plot(
        traj.t,
        radius
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("RMS Radius")

    ax.grid(True)

    ax.set_title(title)

    return ax


# ============================================================
# COMPARISONS
# ============================================================

def compare_phase_space(
    trajectories,
    labels,
    particle=0
):

    n = len(trajectories)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 5)
    )

    if n == 1:
        axes = [axes]

    for ax, traj, label in zip(
        axes,
        trajectories,
        labels
    ):

        phase_space(
            traj,
            particle=particle,
            ax=ax,
            title=label
        )

    plt.tight_layout()

    return fig, axes


# ------------------------------------------------------------

def compare_clouds(
    trajectories,
    labels,
    step=-1,
    size=3
):

    n = len(trajectories)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 5)
    )

    if n == 1:
        axes = [axes]

    for ax, traj, label in zip(
        axes,
        trajectories,
        labels
    ):

        cloud(
            traj,
            step=step,
            ax=ax,
            title=label,
            size=size
        )

    plt.tight_layout()

    return fig, axes


# ------------------------------------------------------------

def compare_energy(
    trajectories,
    labels,
    particle=0
):

    fig, ax = plt.subplots(figsize=(7, 4))

    for traj, label in zip(
        trajectories,
        labels
    ):

        err = analysis.energy_error(traj)

        ax.plot(
            traj.t,
            err[:, particle],
            label=label
        )

    ax.grid(True)

    ax.legend()

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$H-H_0$")

    plt.tight_layout()

    return fig, ax


# ------------------------------------------------------------

def compare_rms_radius(
    trajectories,
    labels
):
    """
    Compare RMS cloud radius.
    """

    fig, ax = plt.subplots(figsize=(7, 4))

    for traj, label in zip(
        trajectories,
        labels
    ):

        radius = analysis.rms_radius_history(traj)

        ax.plot(
            traj.t,
            radius,
            label=label
        )

    ax.grid(True)

    ax.legend()

    ax.set_xlabel("Time")
    ax.set_ylabel("RMS Radius")

    plt.tight_layout()

    return fig, ax