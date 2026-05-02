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

    mean = np.mean(block_means)
    std = np.std(block_means, ddof=1)

    return mean, std


def mean_absolute_spin(magnetization_series, block_size):
    return mean_and_std_from_blocks(magnetization_series, block_size)


def mean_energy_per_spin(energy_per_spin_series, block_size):
    return mean_and_std_from_blocks(energy_per_spin_series, block_size)


def magnetic_susceptibility_per_spin(magnetization_series, beta, n_particles):
    magnetization_series = np.asarray(magnetization_series)
    return beta * n_particles * np.var(magnetization_series)


def specific_heat_per_spin(energy_series, beta, temp, n_particles):
    energy_series = np.asarray(energy_series)
    return beta / (n_particles * temp) * np.var(energy_series)


def block_observable(series, block_size, observable_function, **params):
    blocks = block_series(series, block_size)

    values = [observable_function(block, **params) for block in blocks]

    values = np.asarray(values)

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    return mean, std


def analyze_simulation(loaded_data):
    metadata = loaded_data["metadata"].item()

    magnetization = loaded_data["magnetization_data"]
    energy = loaded_data["energy"]
    energy_per_spin = loaded_data["energy_per_spin"]

    tau = autocorrelation_time(magnetization)
    block_size = max(1, int(16 * tau))

    mean_M, std_M = mean_absolute_spin(magnetization, block_size)
    mean_E, std_E = mean_energy_per_spin(energy_per_spin, block_size)

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

    return {
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
    }
