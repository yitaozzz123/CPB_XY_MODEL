import numpy as np
import matplotlib as plt


class XY_Monte_Carlo:
    def __init__(self, temp, n_particles_1d, external_field=0, all_up=False):
        self.temp = temp
        self.n_particles_1d = n_particles_1d
        self.all_up = all_up
        self.external_field=external_field

        self.n_dim = 2
        self.h_field = 0

        self.n_particles = n_particles_1d**self.n_dim
        self.state = self.initialize_state()
        self.energy = self.hamiltonian()

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


    def hamiltonian(self):
        for i in range(self.n_particles):
            energy = 0
            for j in range(self.n_particles):

                energy += - self.J * np.cos(self.state[i,j]-self.state[i-1,j])
                energy += - self.J * np.cos(self.state[i,j]-self.state[i,j-1])

                field = self.field[i,j]
                field_magnitude = np.sqrt(np.dot(field[i,j],field[i,j]))
                field_angle = np.arctan2(field[i,j])
                field_magnitude * np.cos(self.state[i,j]-field_angle)
        return 0    

    def update_hamiltonian(self, [i,j]):
        return 0

    def current_state():
        return 0
    
    def transition():
        pass

    def initialize_state(self):
        if self.all_up:
            initial_state = self.state_all_up()

        initial_state = self.random_state()
        return initial_state


test = XY_Monte_Carlo(1, 10)
test.random_state()
