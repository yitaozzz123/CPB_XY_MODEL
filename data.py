import numpy as np
from collections import deque
from vortices import count_vortices


class SimulationData:
    def __init__(self, window_size):
        self.window_size = window_size

        self.magnetization_data = []
        self.magnetization_window = deque(maxlen=window_size)
        self.magnetization_moving_average = []

        self.acceptance_ratio = []

        self.energy = []
        self.energy_per_spin = []

        self.n_vortices = []
        self.n_antivortices = []
        self.vortex_density = []

    def store_acceptance(self, acceptance_ratio):
        self.acceptance_ratio.append(acceptance_ratio)

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

    def store_vortices(self, state):

        n_vortices, n_antivortices = count_vortices(state)

        self.n_vortices.append(n_vortices)
        self.n_antivortices.append(n_antivortices)
        self.vortex_density.append((n_vortices + n_antivortices) / state.size)

    def store_step(self, acceptance_ratio, magnetization, energy, n_particles, state):
        self.store_acceptance(acceptance_ratio)
        self.store_magnetization(magnetization)
        self.store_energy(energy, n_particles)
        self.store_vortices(state)
