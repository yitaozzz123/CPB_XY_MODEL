import numpy as np
from itertools import product
from tqdm import tqdm
from thermalization import estimate_thermalization_cut

from model import XY_Monte_Carlo, simulation_metadata
from plots import save_lattice_plot, save_stored_plots
from storage import save_simulation_data
from paths import simulation_data_filename


def make_seed(temp, n_particles_1d):
    return int(temp * 10) + 100 * n_particles_1d


def sweeps_for_configuration(temp, n_particles_1d):

    if temp in [0.5, 0.6, 0.7] and n_particles_1d == 50:
        return 100_000

    if temp in [0.8, 0.9, 0.95] and n_particles_1d == 50:
        return 50_000

    if temp in [0.85, 0.88, 1.0] and n_particles_1d == 50:
        return 20_000

    if temp == 0.5 and n_particles_1d in [10, 20]:
        return 10_000

    if temp == 0.6 and n_particles_1d in [10, 20]:
        return 10_000

    return 5_000


def run_single_simulation(
    temp,
    n_particles_1d,
    external_field_strength=0,
    all_up=False,
):
    n_sweeps = sweeps_for_configuration(temp, n_particles_1d)

    model = XY_Monte_Carlo(
        temp=temp,
        n_particles_1d=n_particles_1d,
        external_field_strength=external_field_strength,
        n_sweeps=n_sweeps,
        seed=make_seed(temp, n_particles_1d),
    )

    save_lattice_plot(model, initial=True)

    data = model.full_transition()
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

    if external_field_strength == 0 and n_particles_1d == 20:
        save_simulation_data(
            filename=simulation_data_filename(model, force_field_folder=True),
            data=data,
            metadata=metadata,
        )


def run_temperature_scan():
    temps = np.arange(0.5, 2.5 + 0.001, 0.2)

    extra_temps = np.array(
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

    temps = np.unique(np.concatenate([temps, extra_temps]))

    particles = [10, 20, 50]

    combinations = list(product(temps, particles))

    for T, N in tqdm(combinations, desc="No-field simulations"):
        run_single_simulation(
            temp=T,
            n_particles_1d=N,
            external_field_strength=0,
        )

    print("Done all no-field simulations")


def run_external_field_scan():
    temps = np.arange(0.5, 2.5 + 0.001, 0.2)

    extra_temps = np.array(
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

    temps = np.unique(np.concatenate([temps, extra_temps]))

    external_fields = [0.25, 0.5, 1, 2]
    n_particles_1d = 20

    combinations = list(product(temps, external_fields))

    for T, h in tqdm(combinations, desc="External-field simulations"):
        run_single_simulation(
            temp=T,
            n_particles_1d=n_particles_1d,
            external_field_strength=h,
        )

    print("Done all external-field simulations")


"""def run_extra_temperature_scan_N50():
    temps = np.array(
        [
            0.50,
            0.60,
            0.70,
            0.80,
            0.85,
            0.88,
            0.90,
            0.95,
            1.00,
        ]
    )

    particles = [50]

    combinations = list(product(temps, particles))

    for T, N in tqdm(combinations, desc="Extra no-field simulations N=50"):
        run_single_simulation(
            temp=T,
            n_particles_1d=N,
            external_field_strength=0,
        )

    print("Done extra no-field simulations N=50")"""
