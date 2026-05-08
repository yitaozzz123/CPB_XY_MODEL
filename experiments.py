"""Experiment runners for XY Monte Carlo simulations."""

import numpy as np
from itertools import product

from tqdm import tqdm
import time

from model import XYMonteCarlo, simulation_metadata
from paths import simulation_data_filename, critical_temperature_study_data_filename
from plots import save_lattice_plot, save_stored_plots
from storage import save_simulation_data
from thermalization import estimate_thermalization_cut


def make_seed(temperature: float, lattice_size: int) -> int:
    """Return a deterministic seed for one simulation configuration."""
    return int(temperature * 10) + 100 * lattice_size


def sweeps_for_configuration(temperature: float, lattice_size: int) -> int:
    """Return the number of sweeps for a simulation configuration."""
    if temperature in [0.5, 0.6, 0.7, 0.85] and lattice_size == 50:
        return 100_000

    if temperature in [0.8, 0.9, 0.95] and lattice_size == 50:
        return 50_000

    if temperature == 0.88 and lattice_size == 50:
        return 200_000

    if temperature in [1.0, 1.1] and lattice_size == 50:
        return 20_000

    if temperature in [0.5, 0.6] and lattice_size in [10, 20]:
        return 10_000

    return 5_000


def temperature_grid() -> np.ndarray:
    """Return the temperatures used in the temperature scans."""
    base_temperatures = np.arange(0.5, 2.5 + 0.001, 0.2)

    extra_temperatures = np.array(
        [
            0.6,
            0.80,
            0.85,
            0.88,
            0.90,
            0.95,
            1.00,
        ]
    )

    return np.unique(np.concatenate([base_temperatures, extra_temperatures]))


def run_single_simulation(
    temperature: float,
    lattice_size: int,
    external_field_strength: float = 0.0,
    all_up: bool = False,
) -> None:
    """Run one simulation, save its plots, and store its data."""
    n_sweeps = sweeps_for_configuration(temperature, lattice_size)

    model = XYMonteCarlo(
        temp=temperature,
        n_particles_1d=lattice_size,
        external_field_strength=external_field_strength,
        n_sweeps=n_sweeps,
        seed=make_seed(temperature, lattice_size),
        all_up=all_up,
    )

    save_lattice_plot(model, initial=True)

    data = model.simulate()
    data.thermalization_cut = estimate_thermalization_cut(
        data.energy_moving_average,
        window_size=data.window_size,
    )

    save_lattice_plot(model, initial=False)
    save_stored_plots(model, data)

    metadata = simulation_metadata(model)

    save_simulation_data(
        filename=simulation_data_filename(model),
        data=data,
        metadata=metadata,
    )

    if external_field_strength == 0 and lattice_size == 20:
        save_simulation_data(
            filename=simulation_data_filename(model, force_field_folder=True),
            data=data,
            metadata=metadata,
        )


def run_temperature_scan() -> None:
    """Run simulations without external field for all temperatures and sizes."""
    temperatures = temperature_grid()
    lattice_sizes = [10, 20, 50]

    configurations = list(product(temperatures, lattice_sizes))

    for temperature, lattice_size in tqdm(configurations, desc="No-field simulations"):
        run_single_simulation(
            temperature=temperature,
            lattice_size=lattice_size,
            external_field_strength=0.0,
        )

    print("Done all no-field simulations")


def run_external_field_scan() -> None:
    """Run simulations with external field for all temperatures and fields."""
    temperatures = temperature_grid()
    external_fields = [0.25, 0.5, 1.0, 2.0]
    lattice_size = 20

    configurations = list(product(temperatures, external_fields))

    for temperature, external_field in tqdm(
        configurations,
        desc="External-field simulations",
    ):
        run_single_simulation(
            temperature=temperature,
            lattice_size=lattice_size,
            external_field_strength=external_field,
        )

    print("Done all external-field simulations")


def run_critical_temperature_rescan() -> None:
    """Run longer simulations near the KT critical region for N=50."""
    configurations = [
        (1.03, 20, 20000),
    ]
    print("Starting")
    for temperature, lattice_size, n_sweeps in configurations:
        print(f"Temperature {temperature}, lattice {lattice_size}, nsweeps {n_sweeps}")
        model = XYMonteCarlo(
            temp=temperature,
            n_particles_1d=lattice_size,
            external_field_strength=0.0,
            n_sweeps=n_sweeps,
            seed=make_seed(temperature, lattice_size),
        )

        save_lattice_plot(model, initial=True)

        data = model.simulate()

        data.thermalization_cut = estimate_thermalization_cut(
            data.energy_moving_average,
            window_size=data.window_size,
        )

        save_lattice_plot(model, initial=False)
        save_stored_plots(model, data)

        metadata = simulation_metadata(model)

        save_simulation_data(
            filename=simulation_data_filename(model),
            data=data,
            metadata=metadata,
        )

    print("Done KT critical-region rescans")


def run_critical_temperature_study_simulations() -> None:
    """Run longer simulations near the KT critical region for N=50."""
    configurations = [
        (temperature, field_strength)
        for temperature in np.linspace(0.6, 1.2, 10)
        for field_strength in np.linspace(0.0, 0.25, 6)
    ]
    print("Starting")
    initial_time = time.time()
    for temperature, field_strenght in configurations:
        print(f"Temperature {temperature}, field {field_strenght}")
        model = XYMonteCarlo(
            temp=temperature,
            n_particles_1d=20,
            external_field_strength=field_strenght,
            n_sweeps=3000,
            seed=make_seed(temperature, 20),
        )

        data = model.simulate()

        data.thermalization_cut = estimate_thermalization_cut(
            data.energy_moving_average,
            window_size=data.window_size,
        )

        metadata = simulation_metadata(model)

        save_simulation_data(
            filename=critical_temperature_study_data_filename(model),
            data=data,
            metadata=metadata,
        )

    print(
        f"simulation done, time spent from the start of the simulation: {time.time()-initial_time}"
    )

    print("Done")
