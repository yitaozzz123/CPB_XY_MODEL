import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from itertools import product
from tqdm import tqdm


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

        self.magnetization_data = []
        self.window_transitions = deque(maxlen=self.n_particles)  # tau
        self.last_transitions_data = []
        self.window_magnetizations_vector = deque(maxlen=self.n_particles)  # tau
        self.last_magnetizations_data = []
        self.last_energy_per_spin = deque(maxlen=self.n_particles)  # tau
        self.last_energy = deque(maxlen=self.n_particles)  # tau

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

        # print(energy_update)

        if self.rng.random() < self.acceptance_probability(energy_update):
            self.state[*particle_index] = new_angle
            self.transition_counter += 1
            self.window_transitions.append(1)
        else:
            self.window_transitions.append(0)

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

    def plot_lattice(self):
        if self.n_dim != 2:
            raise ValueError("Only 2d plot")

        y, x = np.mgrid[0 : self.state.shape[0], 0 : self.state.shape[1]]

        u = np.cos(self.state)
        v = np.sin(self.state)

        self.fig, self.ax = plt.subplots()

        self.q = self.ax.quiver(
            x,
            y,
            u,
            v,
            self.state,
            cmap="twilight",
            norm=plt.Normalize(-np.pi, np.pi),
            scale=40,
            width=0.008,
        )

        self.cbar = self.fig.colorbar(self.q, ax=self.ax, label="Angle (radians)")

        self.ax.set_title("State visualization")
        self.ax.set_xlabel("X index")
        self.ax.set_ylabel("Y index")
        self.ax.set_aspect("equal")

        return self.fig, self.ax

    def save_plot_lattice(self, initial: bool):
        if initial:
            filename = f"{self.base_filename()}_lattice_initial.pdf"
        else:
            filename = f"{self.base_filename()}_lattice_final.pdf"

        self.fig.savefig(filename, bbox_inches="tight")
        plt.close(self.fig)

    # stores magnetizations inside a list:
    #   self.magnetization_data stores the module of the total magnetization for each timestep
    #   self.last_magnetization stores the cartesian vector
    def store_magnetizations(self):
        magnetization = self.total_magnetisation()
        magn_module = np.linalg.norm(magnetization)

        self.magnetization_data.append(magn_module)
        self.window_magnetizations_vector.append(magnetization)

    def full_transition(self):
        for _ in range(self.n_iterations):
            self.single_transition()
            self.store_magnetizations()
            self.store_magnetization_average()
            self.store_transition_ratio()

    def store_transition_ratio(self):
        if len(self.window_transitions) == 100:
            ratio = sum(self.window_transitions) / 100
            self.last_transitions_data.append(ratio)
        # it stores the average of the last 100 timesteps

    def store_magnetization_average(self):
        if len(self.window_magnetizations_vector) == 100:
            avg_magnetization = np.mean(
                np.array(self.window_magnetizations_vector), axis=0
            )
            avg_module_magnetization = np.linalg.norm(avg_magnetization)
            self.last_magnetizations_data.append(avg_module_magnetization)

    def plot_magnetization_data(self, style):
        if len(self.last_magnetizations_data) == 0:
            raise ValueError("No magnetization data to plot.")

        x = np.arange(100, 100 + len(self.last_magnetizations_data))

        fig, ax = plt.subplots(figsize=style["figsize"])
        ax.plot(x, self.last_magnetizations_data, linewidth=style["linewidth"])

        ax.set_title(
            "Magnetization modulus of mean vector", fontsize=style["title_size"]
        )
        ax.set_xlabel("Iteration", fontsize=style["label_size"])
        ax.set_ylabel("|<M>|", fontsize=style["label_size"])
        ax.grid(True, alpha=style["grid_alpha"])
        ax.tick_params(axis="both", labelsize=style["tick_size"])

        fig.tight_layout()
        fig.savefig(
            f"{self.base_filename()}_magnetization.pdf",
            bbox_inches="tight",
        )

        plt.close(fig)

    def plot_transition_data(self, style):
        if len(self.last_transitions_data) == 0:
            raise ValueError("No transition data to plot.")

        x = np.arange(100, 100 + len(self.last_transitions_data))

        fig, ax = plt.subplots(figsize=style["figsize"])
        ax.plot(x, self.last_transitions_data, linewidth=style["linewidth"])

        ax.set_title(
            "Acceptance ratio in the last 100 iterations", fontsize=style["title_size"]
        )
        ax.set_xlabel("Iteration", fontsize=style["label_size"])
        ax.set_ylabel("Ratio", fontsize=style["label_size"])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=style["grid_alpha"])
        ax.tick_params(axis="both", labelsize=style["tick_size"])

        fig.tight_layout()
        fig.savefig(
            f"{self.base_filename()}_transitions.pdf",
            bbox_inches="tight",
        )

        plt.close(fig)

    def plot_stored_data(self):
        style = {
            "figsize": (10, 4),
            "linewidth": 1.8,
            "title_size": 14,
            "label_size": 12,
            "tick_size": 10,
            "grid_alpha": 0.3,
        }

        if len(self.last_magnetizations_data) > 0:
            self.plot_magnetization_data(style)

        if len(self.last_transitions_data) > 0:
            self.plot_transition_data(style)

    def base_filename(self):
        return f"T_{self.temp:04.2f}_N_{self.n_particles_1d:03d}"

    def compute_and_store_energy_per_spin(self):
        self.energy_per_spin = self.energy / self.n_particles
        self.last_energy_per_spin.append(self.energy_per_spin)

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
        test = XY_Monte_Carlo(T, N, n_iterations=1000000)

        test.plot_lattice()
        test.save_plot_lattice(initial=True)

        test.full_transition()

        test.plot_lattice()
        test.save_plot_lattice(initial=False)

        test.plot_stored_data()

    print("Done all simulations for experiment 1")


def experiment_2():  # I use for testing if correlation time works
    test = XY_Monte_Carlo(1, 10, n_iterations=10000)
    # for i in range(1000):
    #    test.single_transition()
    test.full_transition()
    print(len(test.magnetization_data))

    print(test.autocorrelation_time())


experiment_2()
