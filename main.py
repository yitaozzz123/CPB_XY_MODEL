import numpy as np
import matplotlib as plt


class XY_Monte_Carlo:
    def __init__(self, temp, n_particles_1d, all_up=False):
        self.temp = temp
        self.n_particles_1d = n_particles_1d
        self.all_up = all_up

        self.n_dim = 2
        self.h_field = 0

        self.n_particles = n_particles_1d**self.n_dim
        self.state = self.initialize_state()

    def shape(self):
        shape_array = []
        for i in range(self.n_dim):
            shape_array.append(self.n_particles_1d)
        return shape_array

    def random_state(self):
        array = np.random.uniform(-np.pi, np.pi, self.shape())
        return array

    def state_all_up(self):
        state = np.full(self.shape(), np.pi, dtype=np.float32)
        return state

    def transition():

        pass

    def hamiltonian():
        return 0

    def initialize_state(self):
        if self.all_up:
            initial_state = self.state_all_up()

        initial_state = self.random_state()
        return initial_state


test = XY_Monte_Carlo(1, 10)
test.random_state()
