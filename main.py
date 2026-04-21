import numpy as np
import matplotlib.pyplot as plt


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

        self.trasition_counter = 0
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

    def single_transition(self):
        particle_index, new_angle = self.trial_one_spin_change()
        energy_update = self.energy_change(particle_index, new_angle)

        print(energy_update)

        if self.rng.random() < self.acceptance_probability(energy_update):
            self.state[*particle_index] = new_angle
            self.trasition_counter += 1
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

    def update_animation(self):
        u = np.cos(self.state)
        v = np.sin(self.state)

        # aggiorna le frecce
        self.q.set_UVC(u, v)

        # aggiorna i colori associati
        self.q.set_array(self.state.ravel())

        pass

    def animation(self, frames=1000):
        self.plot()

        for _ in range(frames):
            self.single_transition()
            self.update_animation()
            plt.pause()

        self.save_plot()
        return 0

    def save_plot_lattice(self, initial: bool):
        if initial:
            self.fig.savefig(
                f"Plot_{self.n_particles_1d}_part_{self.temp}_temp_initialization.pdf"
            )
        else:
            self.fig.savefig(
                f"Plot_{self.n_particles_1d}_part_{self.temp}_temp_final.pdf"
            )

    def full_transition(self):
        for _ in range(self.n_iterations):
            self.single_transition()


def main():
    temps = np.linspace(0.5, 2.5, 11)
    particles = [10, 20, 50]
    test = XY_Monte_Carlo(1, 10, n_iterations=1000000)

    test.plot_lattice()
    test.save_plot_lattice(initial=True)
    test.full_transition()
    test.plot_lattice()
    test.save_plot_lattice(initial=False)
    print(f"Done {test.n_iterations} iterations")
    print(f"Number of successfull transitions: {test.trasition_counter}")
    print(f"Ratio: {test.trasition_counter/test.n_iterations*100}%")


if __name__ == "__main__":
    main()
