"""Core implementation of the two-dimensional XY Monte Carlo model."""

import numpy as np

from data import SimulationData


class XYMonteCarlo:
    """Two-dimensional XY model evolved with single-spin Metropolis updates."""

    def __init__(
        self,
        temp: float,
        n_particles_1d: int,
        external_field_strength: float = 0.0,
        external_field_angle: float = 0.0,
        all_up: bool = False,
        seed: int | None = None,
        n_sweeps: int = 1000,
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
        self.boltzmann_constant = 1.0
        self.J = 1.0  # XY coupling constant in natural units.

        self.nearest_neighbor_offsets = self.nearest_neighbor_offsets()
        self.state = self.initialize_state()
        self.energy = self.hamiltonian()
        self.beta = 1.0 / (self.boltzmann_constant * self.temp)

    def shape(self) -> tuple[int, int]:
        """Return the lattice shape."""
        return (self.n_particles_1d, self.n_particles_1d)

    def random_state(self) -> np.ndarray:
        """Return a random initial spin configuration."""
        return self.rng.uniform(-np.pi, np.pi, self.shape())

    def state_all_up(self) -> np.ndarray:
        """Return an ordered initial spin configuration."""
        return np.full(self.shape(), np.pi / 2)

    def initialize_state(self) -> np.ndarray:
        """Return the initial lattice state."""
        if self.all_up:
            return self.state_all_up()
        return self.random_state()

    @staticmethod
    def nearest_neighbor_offsets() -> np.ndarray:
        """Return offsets for the four nearest neighbours on a square lattice."""
        return np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=int)

    def hamiltonian(self) -> float:
        """Return the total energy of the current lattice state."""
        energy = 0.0

        for row in range(self.n_particles_1d):
            for column in range(self.n_particles_1d):
                for offset in self.nearest_neighbor_offsets:
                    neighbor_index = np.mod(
                        np.array([row, column]) + offset,
                        self.n_particles_1d,
                    )
                    neighbor_row, neighbor_column = neighbor_index

                    energy += (
                        -0.5
                        * self.J
                        * np.cos(
                            self.state[row, column]
                            - self.state[neighbor_row, neighbor_column]
                        )
                    )

                energy += -self.external_field_strength * np.cos(
                    self.state[row, column] - self.external_field_angle
                )

        return energy

    def propose_spin_update(self) -> tuple[np.ndarray, float]:
        """Return a random lattice index and proposed new spin angle."""
        new_angle = self.rng.uniform(-np.pi, np.pi)
        particle_index = self.rng.integers(0, self.n_particles_1d, self.n_dim)
        return particle_index, new_angle

    def energy_difference(self, particle_index: np.ndarray, new_angle: float) -> float:
        """Return the energy difference caused by one proposed spin update."""
        old_angle = self.state[*particle_index]
        energy_difference = 0.0

        for offset in self.nearest_neighbor_offsets:
            neighbor_index = np.mod(
                particle_index + offset,
                self.n_particles_1d,
            )

            energy_difference -= -self.J * np.cos(
                old_angle - self.state[*neighbor_index]
            )
            energy_difference += -self.J * np.cos(
                new_angle - self.state[*neighbor_index]
            )

        energy_difference -= -self.external_field_strength * np.cos(
            old_angle - self.external_field_angle
        )
        energy_difference += -self.external_field_strength * np.cos(
            new_angle - self.external_field_angle
        )

        return energy_difference

    def acceptance_probability(self, energy_difference: float) -> float:
        """Return the Metropolis acceptance probability."""
        if energy_difference > 0:
            return float(np.exp(-self.beta * energy_difference))
        return 1.0

    def single_transition(self) -> bool:
        """Attempt one single-spin Metropolis-Hastings update."""
        particle_index, new_angle = self.propose_spin_update()
        energy_difference = self.energy_difference(particle_index, new_angle)

        accepted = self.rng.random() < self.acceptance_probability(energy_difference)

        if accepted:
            self.state[*particle_index] = new_angle
            self.energy += energy_difference
            self.transition_counter += 1

        return accepted

    def magnetic_moments_cartesian(self) -> np.ndarray:
        """Return the Cartesian spin vectors for the current lattice state."""
        return np.array([np.cos(self.state), np.sin(self.state)])

    def total_magnetisation(self) -> np.ndarray:
        """Return the mean magnetisation vector."""
        magnetic_moment_vectors = self.magnetic_moments_cartesian()
        return np.mean(magnetic_moment_vectors, axis=(1, 2))

    def sweep(self) -> float:
        """Perform one Monte Carlo sweep and return the acceptance ratio."""
        accepted_count = 0

        for _ in range(self.n_particles):
            accepted_count += int(self.single_transition())

        return accepted_count / self.n_particles

    def simulate(self) -> SimulationData:
        """Run the simulation and return stored observables."""
        data = SimulationData(window_size=100)

        for _ in range(self.n_sweeps):
            acceptance_ratio = self.sweep()
            magnetization = self.total_magnetisation()

            data.store_step(
                acceptance_ratio=acceptance_ratio,
                magnetization=magnetization,
                energy=self.energy,
                n_particles=self.n_particles,
                state=self.state,
            )

        return data


def simulation_metadata(model: XYMonteCarlo) -> dict:
    """Return metadata needed to analyse or reproduce a simulation."""
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
