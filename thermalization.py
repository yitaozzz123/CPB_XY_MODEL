import numpy as np


def estimate_thermalization_cut(
    energy_moving_average,
    window_size=100,
    patience=5,
):
    energy_moving_average = np.asarray(energy_moving_average)

    if len(energy_moving_average) < patience + 2:
        return 0

    differences = np.diff(energy_moving_average)

    for i in range(len(differences) - patience):
        recent = differences[i : i + patience]

        if np.all(recent > 0):
            return i + window_size

    return 0
