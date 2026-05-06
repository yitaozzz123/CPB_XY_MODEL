"""Utilities for estimating the thermalization cut of a simulation."""

import numpy as np


def estimate_thermalization_cut(
    energy_moving_average: np.ndarray | list[float],
    window_size: int = 100,
    patience: int = 5,
) -> int:
    """Estimate the number of initial sweeps to discard.

    The cut is detected from the moving average of the energy. Once the
    moving average increases for `patience` consecutive steps, the simulation
    is considered thermalized and the corresponding sweep index is returned.

    If no thermalization point is detected, return 0.
    """
    energy_moving_average = np.asarray(energy_moving_average)

    if len(energy_moving_average) < patience + 2:
        return 0

    energy_differences = np.diff(energy_moving_average)

    for index in range(len(energy_differences) - patience):
        recent_differences = energy_differences[index : index + patience]

        if np.all(recent_differences > 0):
            return index + window_size

    return 0
