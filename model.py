import numpy as np
from itertools import product
from tqdm import tqdm
from plots import save_lattice_plot, save_stored_plots
from data import SimulationData
from storage import save_simulation_data
from paths import simulation_data_filename
from cross_analysis import (
    analyze_data_folder,
    save_analysis_summary,
    make_standard_plots,
)


class XY_Monte_Carlo:
    def __init__(
        self,
        temp,
        n_particles_1d,
        external_field_strength=0,
        external_field_angle=0,
        all_up=False,
        seed=None,
        n_sweeps=1000,
    ):
        self.temp = temp
        self.n_particles_1d = n_particles_1d
        self.all_up = all_up
        self.external_field_strength = external_field_strength
        self.external_field_angle = external_field_angle
        self.rng = np.random.default_rng(seed=seed)
        self.n_sweeps = n_sweeps

        self.n_dim = 2
        self.n_particles = n_particles_1d**self.n_dim
        self.transition_counter = 0
        self.energy_per_spin = 0
        self.boltzmann_constant = 1

        self.nearest_neighbours = self.define_near_neigh()

        self.J = 1  # natural units, comment on later

        self.state = self.initialize_state()
        self.energy = self.hamiltonian()
        self.beta = 1 / (self.boltzmann_constant * self.temp)

    def shape(self):
        return [self.n_particles_1d] * self.n_dim

    #######################################################
    # State initialization functions

    def random_state(self):
        initial_state = self.rng.uniform(-np.pi, np.pi, self.shape())
        return initial_state

    def state_all_up(self):
        state = np.full(self.shape(), np.pi / 2)
        return state

    def initialize_state(self):
        if self.all_up:
            initial_state = self.state_all_up()
        else:
            initial_state = self.random_state()
        return initial_state

    ###############################################################################
    # Total energy
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

                energy += -self.external_field_strength * np.cos(
                    self.state[i, j] - self.external_field_angle
                )

        return energy

    def define_near_neigh(self):
        return np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=int)

    #########################################################
    """In this set of functions, the transition between two configuration that differs of only one spin is handled
        following the Metropolis-Hastings algorithm:

        - single_transition: compute if the transition happens
        - trial_one_spin_change: decides which element of the lattice 
            we want to change + the value
        - energy change: computes the difference between the two different 
            configuration of spins without computing the complete hamiltonian
        - acceptance probability: computes the probability of the transition following 
            Metropolis-Hastings algorithm
    """

    def single_transition(self):
        particle_index, new_angle = self.trial_one_spin_change()
        energy_update = self.energy_change(particle_index, new_angle)

        accepted = self.rng.random() < self.acceptance_probability(energy_update)

        if accepted:
            self.state[*particle_index] = new_angle
            self.energy += energy_update
            self.transition_counter += 1

        return accepted

    def trial_one_spin_change(self):
        new_angle = self.rng.uniform(-np.pi, np.pi)
        particle_index = self.rng.integers(0, self.n_particles_1d, self.n_dim)

        return particle_index, new_angle

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
            energy -= -self.J * np.cos(old_angle - self.state[*nearest_neighbour_index])
            energy += -self.J * np.cos(new_angle - self.state[*nearest_neighbour_index])

        # field contribution
        energy -= -self.external_field_strength * np.cos(
            old_angle - self.external_field_angle
        )
        energy += -self.external_field_strength * np.cos(
            new_angle - self.external_field_angle
        )

        return energy

    def acceptance_probability(self, energy_update):
        if energy_update > 0:
            acceptance = np.exp(-self.beta * energy_update)
        else:
            acceptance = 1

        return acceptance

    ####################################################################################

    def magnetic_moments_cartesian(self):
        magnetic_moment_vectors = np.array([np.cos(self.state), np.sin(self.state)])
        return magnetic_moment_vectors
        # might want to change how this list is ordered later. Currently it is: n_dim, n_particles_1d, n_particles_1d

    def total_magnetisation(self):
        # Cartesian
        magnetisation_vectors = self.magnetic_moments_cartesian()
        magnetisation_vector = np.mean(magnetisation_vectors, axis=(1, 2))
        return magnetisation_vector

    def sweep_of_transitions(self):
        accepted_count = 0

        for _ in range(self.n_particles):
            accepted = self.single_transition()
            accepted_count += int(accepted)

        return accepted_count / self.n_particles

    def full_transition(self):
        data = SimulationData(window_size=100)

        for _ in range(self.n_sweeps):
            acceptance_ratio = self.sweep_of_transitions()
            magnetization = self.total_magnetisation()

            data.store_step(
                acceptance_ratio=acceptance_ratio,
                magnetization=magnetization,
                energy=self.energy,
                n_particles=self.n_particles,
                state=self.state,
            )

        return data

    def base_filename(self):
        return f"T_{self.temp:04.2f}_N_{self.n_particles_1d:03d}"


def simulation_metadata(model: XY_Monte_Carlo):
    return {
        "temp": model.temp,
        "n_particles_1d": model.n_particles_1d,
        "n_particles": model.n_particles,
        "n_sweeps": model.n_sweeps,
        "beta": model.beta,
        "J": model.J,
        "external_field": model.external_field_strength,
        "all_up": model.all_up,
    }
