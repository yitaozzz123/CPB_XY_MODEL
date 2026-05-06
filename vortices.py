"""Utilities for detecting vortices in a two-dimensional XY spin lattice."""

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle to the interval [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def vortex_map(state: np.ndarray) -> np.ndarray:
    """Return the topological charge of each plaquette in the lattice.

    The charge is obtained by summing wrapped angle differences around each
    elementary square plaquette. A value of +1 indicates a vortex, while -1
    indicates an antivortex.
    """
    angles = state

    angles_right = np.roll(angles, shift=-1, axis=1)
    angles_down = np.roll(angles, shift=-1, axis=0)
    angles_down_right = np.roll(angles_down, shift=-1, axis=1)

    total_angle = (
        wrap_angle(angles_down - angles)
        + wrap_angle(angles_down_right - angles_down)
        + wrap_angle(angles_right - angles_down_right)
        + wrap_angle(angles - angles_right)
    )

    return np.rint(total_angle / (2 * np.pi)).astype(int)


def count_vortices(state: np.ndarray) -> tuple[int, int]:
    """Return the number of vortices and antivortices in the lattice."""
    topological_charges = vortex_map(state)

    n_vortices = int(np.sum(topological_charges == 1))
    n_antivortices = int(np.sum(topological_charges == -1))

    return n_vortices, n_antivortices


def vortex_density(state: np.ndarray) -> float:
    """Return the density of vortices and antivortices per plaquette."""
    n_vortices, n_antivortices = count_vortices(state)
    n_plaquettes = state.shape[0] * state.shape[1]

    return (n_vortices + n_antivortices) / n_plaquettes
