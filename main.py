import numpy as np
import matplotlib as plt

class XY_Monte_Carlo:
    def __init__(self, temp, n_particles, external_field):
        self.temp=temp
        self.n_particles=n_particles
        self.external_field=external_field

        self.energy = self.hamiltonian()

    def shape():
        return 0

    def random_state():
        return 0

    def state_all_up():
        return 0
    
    def transition():

        pass

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


    def current_state():
        return 0
    


