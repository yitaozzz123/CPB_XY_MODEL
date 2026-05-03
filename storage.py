import numpy as np


def save_simulation_data(filename, data, metadata):
    np.savez(
        filename,
        magnetization_data=np.array(data.magnetization_data),
        magnetization_moving_average=np.array(data.magnetization_moving_average),
        acceptance_ratio=np.array(data.acceptance_ratio),
        energy=np.array(data.energy),
        energy_per_spin=np.array(data.energy_per_spin),
        n_vortices=np.array(data.n_vortices),
        n_antivortices=np.array(data.n_antivortices),
        vortex_density=np.array(data.vortex_density),
        metadata=np.array(metadata, dtype=object),
    )


def load_simulation_data(filename):
    return np.load(filename, allow_pickle=True)
