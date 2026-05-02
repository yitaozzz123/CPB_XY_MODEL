import numpy as np
from collections import deque


class SimulationData:
    def __init__(self, window_size):
        self.window_size = window_size

        self.magnetization_data = []
        self.magnetization_window = deque(maxlen=window_size)
        self.magnetization_moving_average = []

        self.acceptance_window = deque(maxlen=window_size)
        self.acceptance_ratio = []

        self.energy = []
        self.energy_per_spin = []

    def store_acceptance(self, accepted):
        self.acceptance_window.append(int(accepted))

        if len(self.acceptance_window) == self.window_size:
            ratio = sum(self.acceptance_window) / self.window_size
            self.acceptance_ratio.append(ratio)

    def store_magnetization(self, magnetization):
        magnetization_module = np.linalg.norm(magnetization)

        self.magnetization_data.append(magnetization_module)
        self.magnetization_window.append(magnetization)

        if len(self.magnetization_window) == self.window_size:
            avg_magnetization = np.mean(np.array(self.magnetization_window), axis=0)
            avg_module = np.linalg.norm(avg_magnetization)
            self.magnetization_moving_average.append(avg_module)

    def store_energy(self, energy, n_particles):
        self.energy.append(energy)
        self.energy_per_spin.append(energy / n_particles)

    def store_step(self, accepted, magnetization, energy, n_particles):
        self.store_acceptance(accepted)
        self.store_magnetization(magnetization)
        self.store_energy(energy, n_particles)
