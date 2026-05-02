import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from itertools import product
from tqdm import tqdm
from plots import save_lattice_plot, save_stored_plots
from data import SimulationData
from storage import save_simulation_data
from paths import simulation_data_filename


class XY_Monte_Carlo:
    def __init__(
        self,
        temp,
        n_particles_1d,
        external_field=0,
        all_up=False,
        seed=None,
        n_iterations=1000,
    ):
        self.temp = temp
        self.n_particles_1d = n_particles_1d
        self.all_up = all_up
        self.external_field = external_field
        self.rng = np.random.default_rng(seed=seed)
        self.n_iterations = n_iterations

        self.n_dim = 2
        self.n_particles = n_particles_1d**self.n_dim
        self.transition_counter = 0
        self.energy_per_spin = 0
        self.h_field = 0
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

                """
                # external field contribution
                field = self.field[i, j]
                field_magnitude = np.sqrt(np.dot(field, field))
                field_angle = np.arctan2(field)
                energy += field_magnitude * np.cos(self.state[i, j] - field_angle)
                """
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

    def full_transition(self):
        data = SimulationData(window_size=self.n_particles)

        for _ in range(self.n_iterations):
            accepted = self.single_transition()
            magnetization = self.total_magnetisation()

            data.store_step(
                accepted=accepted,
                magnetization=magnetization,
                energy=self.energy,
                n_particles=self.n_particles,
            )

        return data

    def base_filename(self):
        return f"T_{self.temp:04.2f}_N_{self.n_particles_1d:03d}"

    def autocorrelation(self, sweep_index):
        magnetisations = np.array(self.magnetization_data)
        number_sweeps = len(magnetisations) - sweep_index
        autocorrelation = (
            np.trapezoid(magnetisations[:number_sweeps] * magnetisations[sweep_index:])
            / number_sweeps
        )
        autocorrelation -= (
            np.trapezoid(magnetisations[:number_sweeps])
            * np.trapezoid(magnetisations[sweep_index:])
            * number_sweeps**-2
        )
        return autocorrelation

    def autocorrelation_time(self):
        autocorrelation0 = self.autocorrelation(0)
        autocorrelation = autocorrelation0
        correlation_time = 0
        sweep_index = 0
        sweep_max = len(self.magnetization_data)
        # print(sweep_max, "sweep max")
        while (autocorrelation > 0) and (sweep_index < sweep_max):
            autocorrelation = self.autocorrelation(sweep_index)
            sweep_index += 1
            correlation_time += autocorrelation / autocorrelation0
            # print(autocorrelation, sweep_index)
        return correlation_time

    def compute_chi_M(self):
        var_M = np.var(self.last_magnetizations)
        chi_M = self.beta * var_M / self.n_particles
        return chi_M

    def compute_C(self):
        var_E = np.var(self.last_energy)
        C = self.beta * var_E / (self.n_particles * self.temp)
        return C


def experiment_1():
    temps = np.linspace(0.5, 2.5, 11)
    particles = [10, 20, 50]

    combinations = list(product(temps, particles))

    for T, N in tqdm(combinations, desc="Simulations"):
        model = XY_Monte_Carlo(T, N, n_iterations=1000000)

        save_lattice_plot(model, initial=True)

        data = model.full_transition()

        save_lattice_plot(model, initial=False)
        save_stored_plots(model, data)

        metadata = {
            "temp": model.temp,
            "n_particles_1d": model.n_particles_1d,
            "n_particles": model.n_particles,
            "n_iterations": model.n_iterations,
            "beta": model.beta,
            "J": model.J,
            "external_field": model.external_field,
            "all_up": model.all_up,
        }

        save_simulation_data(
            filename=simulation_data_filename(model),
            data=data,
            metadata=metadata,
        )

    print("Done all simulations for experiment 1")


def experiment_2():  # I use for testing if correlation time works
    test = XY_Monte_Carlo(1, 10, n_iterations=10000)
    # for i in range(1000):
    #    test.single_transition()
    test.full_transition()
    print(len(test.magnetization_data))

    print(test.autocorrelation_time())


def experiment_3():
    model = XY_Monte_Carlo(0.8, 30)

    save_lattice_plot(model, initial=True)

    data = model.full_transition()

    save_lattice_plot(model, initial=False)
    save_stored_plots(model, data)
    metadata = {
        "temp": model.temp,
        "n_particles_1d": model.n_particles_1d,
        "n_particles": model.n_particles,
        "n_iterations": model.n_iterations,
        "beta": model.beta,
        "J": model.J,
        "external_field": model.external_field,
        "all_up": model.all_up,
    }
    save_simulation_data(
        filename=simulation_data_filename(model), data=data, metadata=metadata
    )


experiment_3()
