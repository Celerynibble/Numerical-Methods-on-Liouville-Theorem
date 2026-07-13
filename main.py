"""
main.py
========

Demonstration of the HamiltonianFlow library.

Experiment 1
------------
Euler vs RK4
Short integration
Single trajectory

Experiment 2
------------
RK4 vs Störmer-Verlet
Long integration
Single trajectory

Experiment 3
------------
Cloud evolution
RK4 vs Störmer-Verlet
Three snapshots
"""

import numpy as np
import matplotlib.pyplot as plt

from hamiltonian import Pendulum
from cloud import Cloud
from integrators import Euler, RK4, StormerVerlet
import plotting

from shapely.geometry import Polygon, MultiPolygon
import analysis


# ============================================================
# SYSTEM
# ============================================================

system = Pendulum()


# ============================================================
# EXPERIMENT 1
# Euler vs RK4
# ============================================================

dt = 0.1
T = 40

cloud = Cloud.single_particle(
    q=np.pi/2,
    p=0
)

euler = Euler(system, dt)
rk4 = RK4(system, dt)

traj_euler = euler.evolve(cloud, T)
traj_rk4 = rk4.evolve(cloud, T)

plotting.compare_phase_space(
    [traj_euler, traj_rk4],
    ["Euler", "RK4"]
)

plotting.compare_energy(
    [traj_euler, traj_rk4],
    ["Euler", "RK4"]
)


# ============================================================
# EXPERIMENT 2
# RK4 vs Verlet
# ============================================================

dt = 0.5
T = 3000

cloud = Cloud.single_particle(
    q=np.pi/2,
    p=0
)

rk4 = RK4(system, dt)
verlet = StormerVerlet(system, dt)

traj_rk4 = rk4.evolve(cloud, T)
traj_verlet = verlet.evolve(cloud, T)

plotting.compare_phase_space(
    [traj_rk4, traj_verlet],
    ["RK4", "Störmer-Verlet"]
)

plotting.compare_energy(
    [traj_rk4, traj_verlet],
    ["RK4", "Störmer-Verlet"]
)


# ============================================================
# EXPERIMENT 3
# Liouville theorem
# ============================================================

dt = 0.5
T = 400

cloud = Cloud.disk(
    center=(np.pi/6, 0),
    radius=0.03,
    N=8000          # increase this until the blob looks continuous
)

rk4 = RK4(system, dt)
verlet = StormerVerlet(system, dt)

traj_rk4 = rk4.evolve(cloud, T)
traj_verlet = verlet.evolve(cloud, T)


# ------------------------------------------------------------
# Snapshots
# ------------------------------------------------------------

steps = [
    0,
    traj_rk4.nsteps // 2,
    traj_rk4.nsteps - 1
]

labels = [
    "Initial",
    "Middle",
    "Final"
]

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green"
]


# ------------------------------------------------------------
# Common limits
# ------------------------------------------------------------

all_q = np.concatenate((traj_rk4.q.ravel(),
                        traj_verlet.q.ravel()))

all_p = np.concatenate((traj_rk4.p.ravel(),
                        traj_verlet.p.ravel()))

margin = 0.1

xmin = all_q.min() - margin
xmax = all_q.max() + margin

ymin = all_p.min() - margin
ymax = all_p.max() + margin


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))


plotting.cloud_snapshots(
    traj_rk4,
    steps=steps,
    labels=labels,
    colors=colors,
    ax=axes[0],
    size=0.6,
    alpha=0.18,
    title="RK4"
)

plotting.cloud_snapshots(
    traj_verlet,
    steps=steps,
    labels=labels,
    colors=colors,
    ax=axes[1],
    size=0.6,
    alpha=0.18,
    title="Störmer-Verlet"
)


for ax in axes:

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")


plt.tight_layout()
plt.show()