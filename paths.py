from pathlib import Path


def has_external_field(model):
    return model.external_field is not None and model.external_field != 0


def results_folder(model):
    if has_external_field(model):
        return Path("results_with_field")
    return Path("results")


def data_folder(model):
    if has_external_field(model):
        return Path("data_with_field")
    return Path("data")


def base_filename(model):
    name = f"T_{model.temp:04.2f}_N_{model.n_particles_1d:03d}"

    if has_external_field(model):
        name += f"_h_{model.external_field:04.2f}"

    return name


def lattice_filename(model, initial: bool):
    suffix = "lattice_initial" if initial else "lattice_final"
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{base_filename(model)}_{suffix}.pdf"


def magnetization_filename(model):
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{base_filename(model)}_magnetization.pdf"


def transitions_filename(model):
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{base_filename(model)}_transitions.pdf"


def simulation_data_filename(model):
    folder = data_folder(model)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{base_filename(model)}_data.npz"


def energy_filename(model):
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{base_filename(model)}_energy_per_spin.pdf"


def vortex_filename(model):
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{base_filename(model)}_vortex_density.pdf"


def vortex_count_filename(model):
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{base_filename(model)}_vortex_counts.pdf"
