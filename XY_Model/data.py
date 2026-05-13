"""Containers for observables collected during one XY Monte Carlo simulation."""

from collections import deque

import numpy as np

from vortices import count_vortices


class SimulationData:
    """Store observables measured during a Monte Carlo simulation."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        self.magnetization_data = []
        self.magnetization_window = deque(maxlen=window_size)
        self.magnetization_moving_average = []

        self.energy = []
        self.energy_per_spin = []
        self.energy_window = deque(maxlen=window_size)
        self.energy_moving_average = []

        self.acceptance_ratio = []

        self.n_vortices = []
        self.n_antivortices = []
        self.vortex_density = []

        self.thermalization_cut = 0

    def store_acceptance(self, acceptance_ratio: float) -> None:
        """Store the acceptance ratio of one Monte Carlo sweep."""
        self.acceptance_ratio.append(acceptance_ratio)

    def store_magnetization(self, magnetization: np.ndarray) -> None:
        """Store magnetization and its moving average."""
        magnetization_module = np.linalg.norm(magnetization)

        self.magnetization_data.append(magnetization_module)
        self.magnetization_window.append(magnetization)

        if len(self.magnetization_window) == self.window_size:
            average_magnetization = np.mean(
                np.array(self.magnetization_window),
                axis=0,
            )
            average_module = np.linalg.norm(average_magnetization)
            self.magnetization_moving_average.append(average_module)

    def store_energy(self, energy: float, n_particles: int) -> None:
        """Store total energy, energy per spin, and moving average."""
        energy_per_spin = energy / n_particles

        self.energy.append(energy)
        self.energy_per_spin.append(energy_per_spin)

        self.energy_window.append(energy_per_spin)

        if len(self.energy_window) == self.window_size:
            self.energy_moving_average.append(np.mean(self.energy_window))

    def store_vortices(self, state: np.ndarray) -> None:
        """Store vortex and antivortex counts for one lattice state."""
        n_vortices, n_antivortices = count_vortices(state)

        self.n_vortices.append(n_vortices)
        self.n_antivortices.append(n_antivortices)
        self.vortex_density.append((n_vortices + n_antivortices) / state.size)

    def store_step(
        self,
        acceptance_ratio: float,
        magnetization: np.ndarray,
        energy: float,
        n_particles: int,
        state: np.ndarray,
    ) -> None:
        """Store all observables measured after one Monte Carlo sweep."""
        self.store_acceptance(acceptance_ratio)
        self.store_magnetization(magnetization)
        self.store_energy(energy, n_particles)
        self.store_vortices(state)
