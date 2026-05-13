"""Input/output utilities for storing and loading simulation data."""

from pathlib import Path

import numpy as np

from data import SimulationData


def save_simulation_data(
    filename: str | Path,
    data: SimulationData,
    metadata: dict,
) -> None:
    """Save simulation observables and metadata to a compressed NumPy file."""
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
        energy_moving_average=np.array(data.energy_moving_average),
        thermalization_cut=np.array(data.thermalization_cut),
    )


def load_simulation_data(filename: str | Path) -> np.lib.npyio.NpzFile:
    """Load simulation data from a NumPy archive."""
    return np.load(filename, allow_pickle=True)
