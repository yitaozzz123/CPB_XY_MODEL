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

        self.nearest_neighbours = np.array(
            [[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=int
        )
        self.J = 1  # natural units, comment on later

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
        energy = 0
        # loop over all particles in the x and y direction
        for i in range(self.n_particles_1d):
            for j in range(self.n_particles_1d):

                # nearest neighbour contribution
                # loop over nearest neighbour
                for k in range(len(self.nearest_neighbours)):
                    nearest_neighbour_index = np.mod(
                        np.array([i, j]) + self.nearest_neighbours[k],
                        self.n_particles_1d,
                    )
                    energy += (
                        -0.5
                        * self.J
                        * np.cos(
                            self.state[i, j] - self.state[*nearest_neighbour_index]
                        )
                    )

                """
                # external field contribution
                field = self.field[i, j]
                field_magnitude = np.sqrt(np.dot(field, field))
                field_angle = np.arctan2(field)
                energy += field_magnitude * np.cos(self.state[i, j] - field_angle)
                """
        return energy

    def energy_change(self, particle_index, new_angle):

        old_angle = self.state[*particle_index]
        energy = 0

        # nearest neighbour contribution
        # loop over nearest neighbour
        for k in range(len(self.nearest_neighbours)):
            nearest_neighbour_index = np.mod(
                np.array(particle_index) + self.nearest_neighbours[k],
                self.n_particles_1d,
            )
            energy -= (
                -0.5 * self.J * np.cos(old_angle - self.state[*nearest_neighbour_index])
            )
            energy += (
                -0.5 * self.J * np.cos(new_angle - self.state[*nearest_neighbour_index])
            )

        # field contribution
        """
        field = self.field[*particle_index]
        field_magnitude = np.sqrt(np.dot(field, field))
        field_angle = np.arctan2(field)

        energy -= field_magnitude * np.cos(old_angle - field_angle)
        energy += field_magnitude * np.cos(new_angle - field_angle)
        """

        return energy

    def current_state():
        return 0

    def acceptance_probability(self, energy_update):
        if energy_update > 0:
            acceptance = np.exp(-self.beta * energy_update)
        else:
            acceptance = 1

        return acceptance

    def transition(self):
        particle_index, new_angle = self.trial_one_spin_change()
        energy_update = self.energy_change(particle_index, new_angle)

        print(energy_update)

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
