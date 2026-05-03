import numpy as np


def autocorrelation(series, lag):
    series = np.asarray(series)
    n = len(series) - lag

    if n <= 0:
        raise ValueError("lag is too large for this series")

    x = series[:n]
    y = series[lag:]

    return np.mean(x * y) - np.mean(x) * np.mean(y)


def autocorrelation_time(magnetization_series):
    c0 = autocorrelation(magnetization_series, 0)

    if c0 == 0:
        return 0.0

    tau = 0.0

    for lag in range(len(magnetization_series)):
        c_lag = autocorrelation(magnetization_series, lag)

        if c_lag < 0:
            break

        tau += c_lag / c0

    return tau


def block_series(series, block_size):
    series = np.asarray(series)
    n_blocks = len(series) // block_size

    if n_blocks == 0:
        raise ValueError("block_size is larger than the data series")

    trimmed = series[: n_blocks * block_size]
    return trimmed.reshape(n_blocks, block_size)


def mean_and_std_from_blocks(series, block_size):
    blocks = block_series(series, block_size)
    block_means = np.mean(blocks, axis=1)

    return np.mean(block_means), np.std(block_means, ddof=1)


def magnetic_susceptibility_per_spin(magnetization_series, beta, n_particles):
    magnetization_series = np.asarray(magnetization_series)
    return beta * n_particles * np.var(magnetization_series)


def specific_heat_per_spin(energy_series, beta, temp, n_particles):
    energy_series = np.asarray(energy_series)
    return beta / (n_particles * temp) * np.var(energy_series)


def block_observable(series, block_size, observable_function, **params):
    blocks = block_series(series, block_size)

    values = np.array([observable_function(block, **params) for block in blocks])

    return np.mean(values), np.std(values, ddof=1)


def analyze_simulation(loaded_data):
    metadata = loaded_data["metadata"].item()

    magnetization = loaded_data["magnetization_data"]
    energy = loaded_data["energy"]
    energy_per_spin = loaded_data["energy_per_spin"]
    vortex_density = loaded_data["vortex_density"]
    n_vortices = loaded_data["n_vortices"]
    n_antivortices = loaded_data["n_antivortices"]

    tau = autocorrelation_time(magnetization)
    block_size = max(1, int(16 * tau))

    mean_M, std_M = mean_and_std_from_blocks(magnetization, block_size)
    mean_E, std_E = mean_and_std_from_blocks(energy_per_spin, block_size)

    chi_M, std_chi_M = block_observable(
        magnetization,
        block_size,
        magnetic_susceptibility_per_spin,
        beta=metadata["beta"],
        n_particles=metadata["n_particles"],
    )

    C, std_C = block_observable(
        energy,
        block_size,
        specific_heat_per_spin,
        beta=metadata["beta"],
        temp=metadata["temp"],
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
        "mean_absolute_spin": mean_M,
        "std_mean_absolute_spin": std_M,
        "energy_per_spin": mean_E,
        "std_energy_per_spin": std_E,
        "magnetic_susceptibility_per_spin": chi_M,
        "std_magnetic_susceptibility_per_spin": std_chi_M,
        "specific_heat_per_spin": C,
        "std_specific_heat_per_spin": std_C,
        "mean_vortex_density": mean_vortex_density,
        "std_vortex_density": std_vortex_density,
        "mean_n_vortices": mean_n_vortices,
        "std_n_vortices": std_n_vortices,
        "mean_n_antivortices": mean_n_antivortices,
        "std_n_antivortices": std_n_antivortices,
    }
