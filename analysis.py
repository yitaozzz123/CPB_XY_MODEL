"""Analysis utilities for XY Monte Carlo simulation data."""

import numpy as np


def autocorrelation(series: np.ndarray | list[float], lag: int) -> float:
    """Return the autocorrelation of a time series at a given lag."""
    series = np.asarray(series)
    n_samples = len(series) - lag

    if n_samples <= 0:
        raise ValueError("lag is too large for this series")

    first_series = series[:n_samples]
    shifted_series = series[lag:]

    return float(
        np.mean(first_series * shifted_series)
        - np.mean(first_series) * np.mean(shifted_series)
    )


def autocorrelation_time(magnetization_series: np.ndarray | list[float]) -> float:
    """Estimate the integrated autocorrelation time of a magnetization series."""
    reference_autocorrelation = autocorrelation(magnetization_series, 0)

    if reference_autocorrelation == 0:
        return 0.0

    tau = 0.0

    for lag in range(len(magnetization_series)):
        current_autocorrelation = autocorrelation(magnetization_series, lag)

        if current_autocorrelation < 0:
            break

        tau += current_autocorrelation / reference_autocorrelation

    return float(tau)


def block_series(
    series: np.ndarray | list[float],
    block_size: int,
) -> np.ndarray:
    """Split a time series into equal-size blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    series = np.asarray(series)
    n_blocks = len(series) // block_size

    if n_blocks == 0:
        raise ValueError("block_size is larger than the data series")

    trimmed_series = series[: n_blocks * block_size]

    return trimmed_series.reshape(n_blocks, block_size)


def mean_and_std_with_tau(
    series: np.ndarray | list[float],
    tau: float,
) -> tuple[float, float]:
    """Return the mean and autocorrelation-corrected standard error."""
    series = np.asarray(series)

    mean = np.mean(series)
    variance = np.var(series)
    n_samples = len(series)

    standard_error = np.sqrt(2 * tau / n_samples * variance)

    return float(mean), float(standard_error)


def mean_and_std_from_blocks(
    series: np.ndarray | list[float],
    block_size: int,
) -> tuple[float, float]:
    """Return the mean and standard deviation of block averages."""
    blocks = block_series(series, block_size)
    block_means = np.mean(blocks, axis=1)

    return float(np.mean(block_means)), float(np.std(block_means, ddof=1))


def magnetic_susceptibility_per_spin(
    magnetization_series: np.ndarray | list[float],
    beta: float,
    n_particles: int,
) -> float:
    """Return the magnetic susceptibility per spin."""
    magnetization_series = np.asarray(magnetization_series)

    return float(beta * n_particles * np.var(magnetization_series))


def specific_heat_per_spin(
    energy_series: np.ndarray | list[float],
    beta: float,
    temperature: float,
    n_particles: int,
) -> float:
    """Return the specific heat per spin."""
    energy_series = np.asarray(energy_series)

    return float(beta / (n_particles * temperature) * np.var(energy_series))


def block_observable(
    series: np.ndarray | list[float],
    block_size: int,
    observable_function,
    **parameters,
) -> tuple[float, float]:
    """Return the mean and standard deviation of a block observable."""
    blocks = block_series(series, block_size)

    observable_values = np.array(
        [observable_function(block, **parameters) for block in blocks]
    )

    return float(np.mean(observable_values)), float(np.std(observable_values, ddof=1))


def analyze_simulation(loaded_data) -> dict:
    """Analyse one loaded simulation archive and return summary observables."""
    thermalization_cut = int(loaded_data.get("thermalization_cut", 0))
    metadata = loaded_data["metadata"].item()

    magnetization = loaded_data["magnetization_data"][thermalization_cut:]
    energy = loaded_data["energy"][thermalization_cut:]
    energy_per_spin = loaded_data["energy_per_spin"][thermalization_cut:]
    vortex_density = loaded_data["vortex_density"][thermalization_cut:]
    n_vortices = loaded_data["n_vortices"][thermalization_cut:]
    n_antivortices = loaded_data["n_antivortices"][thermalization_cut:]

    tau = autocorrelation_time(magnetization)
    block_size = max(1, int(16 * tau))

    mean_magnetization, std_magnetization = mean_and_std_with_tau(
        magnetization,
        tau,
    )
    mean_energy, std_energy = mean_and_std_with_tau(
        energy_per_spin,
        tau,
    )

    susceptibility, std_susceptibility = block_observable(
        magnetization,
        block_size,
        magnetic_susceptibility_per_spin,
        beta=metadata["beta"],
        n_particles=metadata["n_particles"],
    )

    specific_heat, std_specific_heat = block_observable(
        energy,
        block_size,
        specific_heat_per_spin,
        beta=metadata["beta"],
        temperature=metadata["temp"],
        n_particles=metadata["n_particles"],
    )

    mean_vortex_density, std_vortex_density = mean_and_std_from_blocks(
        vortex_density,
        block_size,
    )

    mean_n_vortices, std_n_vortices = mean_and_std_from_blocks(
        n_vortices,
        block_size,
    )

    mean_n_antivortices, std_n_antivortices = mean_and_std_from_blocks(
        n_antivortices,
        block_size,
    )

    return {
        "temp": metadata["temp"],
        "external_field": metadata["external_field"],
        "n_particles_1d": metadata["n_particles_1d"],
        "n_particles": metadata["n_particles"],
        "n_sweeps": metadata["n_sweeps"],
        "beta": metadata["beta"],
        "J": metadata["J"],
        "tau": tau,
        "block_size": block_size,
        "mean_absolute_spin": mean_magnetization,
        "std_mean_absolute_spin": std_magnetization,
        "energy_per_spin": mean_energy,
        "std_energy_per_spin": std_energy,
        "magnetic_susceptibility_per_spin": susceptibility,
        "std_magnetic_susceptibility_per_spin": std_susceptibility,
        "specific_heat_per_spin": specific_heat,
        "std_specific_heat_per_spin": std_specific_heat,
        "mean_vortex_density": mean_vortex_density,
        "std_vortex_density": std_vortex_density,
        "mean_n_vortices": mean_n_vortices,
        "std_n_vortices": std_n_vortices,
        "mean_n_antivortices": mean_n_antivortices,
        "std_n_antivortices": std_n_antivortices,
        "thermalization_cut": thermalization_cut,
        "n_samples_after_thermalization": len(magnetization),
    }
