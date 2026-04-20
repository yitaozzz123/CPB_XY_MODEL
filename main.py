import numpy as np
import matplotlib as plt


class XY_Monte_Carlo:
    def __init__(self, temp, n_particles_1d, external_field=0, all_up=False, seed=None):
        self.temp = temp
        self.n_particles_1d = n_particles_1d
        self.all_up = all_up
        self.external_field = external_field
        self.rng = np.random.default_rng(seed=seed)

        self.n_dim = 2
        self.h_field = 0
        self.boltzmann_constant = 1

        self.n_particles = n_particles_1d**self.n_dim
        self.state = self.initialize_state()
        self.energy = self.hamiltonian()
        self.beta = 1 / (self.boltzmann_constant * self.temp)

    def shape(self):
        shape_array = []
        for i in range(self.n_dim):
            shape_array.append(self.n_particles_1d)
        return shape_array

    def random_state(self):
        initial_state = self.rng.uniform(-np.pi, np.pi, self.shape())
        return initial_state

    def state_all_up(self):
        state = np.full(self.shape(), np.pi / 2)
        return state

    def hamiltonian(self):
        """for i in range(self.n_particles):
        energy = 0
        for j in range(self.n_particles):

            energy += -self.J * np.cos(self.state[i, j] - self.state[i - 1, j])
            energy += -self.J * np.cos(self.state[i, j] - self.state[i, j - 1])

            field = self.field[i, j]
            field_magnitude = np.sqrt(np.dot(field[i, j], field[i, j]))
            field_angle = np.arctan2(field[i, j])
            field_magnitude * np.cos(self.state[i, j] - field_angle)"""
        return 0

    def update_hamiltonian(self, particle_index):
        return 0

    def current_state():
        return 0

    def acceptance_probability(self, energy_update):
        if energy_update > 0:
            acceptance = np.exp ** (-self.beta * self.energy_update)
        else:
            acceptance = 1

        return acceptance

    def transition(self):
        particle_index, new_angle = self.trial_one_spin_change()
        # energy_update = self.energy_change(particle_index, new_angle)

        energy_update = -1

        if self.rng.random() < self.acceptance_probability(energy_update):
            self.state[*particle_index] = new_angle
        pass

    def initialize_state(self):
        if self.all_up:
            initial_state = self.state_all_up()
        else:
            initial_state = self.random_state()
        return initial_state

    def trial_one_spin_change(self):
        new_angle = self.rng.uniform(-np.pi, np.pi)
        particle_index = self.rng.integers(0, self.n_particles_1d, self.n_dim)

        return particle_index, new_angle


test = XY_Monte_Carlo(1, 10)
a = test.state.copy()
test.transition()
b = test.state.copy()

print(a - b)
