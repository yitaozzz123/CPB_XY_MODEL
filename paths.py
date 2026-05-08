"""Utilities for constructing output paths and filenames."""

from pathlib import Path


def has_external_field(model) -> bool:
    """Return whether the model uses a non-zero external field."""
    return (
        model.external_field_strength is not None and model.external_field_strength != 0
    )


def results_folder(model) -> Path:
    """Return the folder used for simulation plots."""
    if has_external_field(model):
        return Path("results_with_field")

    return Path("results")


def data_folder(
    model,
    force_field_folder: bool = False,
) -> Path:
    """Return the folder used for simulation data files."""
    if force_field_folder or has_external_field(model):
        return Path("data_with_field")

    return Path("data")


def base_filename(model) -> str:
    """Return the standard filename prefix for a simulation."""
    filename = f"T_{model.temp:04.2f}" f"_N_{model.n_particles_1d:03d}"

    if has_external_field(model):
        filename += f"_h_{model.external_field_strength:04.2f}"

    return filename


def lattice_filename(model, initial: bool) -> Path:
    """Return the filename for a lattice configuration plot."""
    suffix = "lattice_initial" if initial else "lattice_final"

    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)

    return folder / f"{base_filename(model)}_{suffix}.pdf"


def magnetization_filename(model) -> Path:
    """Return the filename for the magnetization plot."""
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)

    return folder / f"{base_filename(model)}_magnetization.pdf"


def transitions_filename(model) -> Path:
    """Return the filename for the acceptance-ratio plot."""
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)

    return folder / f"{base_filename(model)}_transitions.pdf"


def energy_filename(model) -> Path:
    """Return the filename for the energy plot."""
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)

    return folder / f"{base_filename(model)}_energy_per_spin.pdf"


def vortex_filename(model) -> Path:
    """Return the filename for the vortex-density plot."""
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)

    return folder / f"{base_filename(model)}_vortex_density.pdf"


def vortex_count_filename(model) -> Path:
    """Return the filename for the vortex-count plot."""
    folder = results_folder(model)
    folder.mkdir(parents=True, exist_ok=True)

    return folder / f"{base_filename(model)}_vortex_counts.pdf"


def simulation_data_filename(
    model,
    force_field_folder: bool = False,
) -> Path:
    """Return the filename used to store simulation data."""
    folder = data_folder(
        model,
        force_field_folder=force_field_folder,
    )

    folder.mkdir(parents=True, exist_ok=True)

    filename = base_filename(model)

    if force_field_folder and not has_external_field(model):
        filename += "_h_0.00"

    return folder / f"{filename}_data.npz"


def critical_temperature_study_data_filename(
    model,
    force_field_folder: bool = False,
) -> Path:
    """Return the filename used to store critical temperature study simulation data."""

    folder = (
        data_folder(
            model,
            force_field_folder=force_field_folder,
        )
        / "tau_study"
    )

    folder.mkdir(parents=True, exist_ok=True)

    filename = base_filename(model)

    if force_field_folder and not has_external_field(model):
        filename += "_h_0.00"

    return folder / f"{filename}_data.npz"
