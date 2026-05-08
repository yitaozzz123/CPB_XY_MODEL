from pathlib import Path
from types import SimpleNamespace

from model import XYMonteCarlo
from paths import base_filename, results_folder
from plots import make_vortex_count_figure
from storage import load_simulation_data


def model_from_metadata(metadata: dict) -> XYMonteCarlo:
    return XYMonteCarlo(
        temp=metadata["temp"],
        n_particles_1d=metadata["n_particles_1d"],
        external_field_strength=metadata["external_field"],
        n_sweeps=metadata["n_sweeps"],
    )


def data_from_npz(loaded):
    return SimpleNamespace(
        n_vortices=loaded["n_vortices"],
        n_antivortices=loaded["n_antivortices"],
        vortex_density=loaded["vortex_density"],
    )


def regenerate_folder(data_folder: str) -> None:
    for filename in sorted(Path(data_folder).glob("*_data.npz")):
        loaded = load_simulation_data(filename)
        metadata = loaded["metadata"].item()

        model = model_from_metadata(metadata)
        data = data_from_npz(loaded)

        fig, _ = make_vortex_count_figure(data)
        fig.savefig(
            results_folder(model) / f"{base_filename(model)}_vortex_counts.pdf",
            bbox_inches="tight",
        )

        print(f"Regenerated vortex count plot for {filename}")


def main() -> None:
    regenerate_folder("data")
    # regenerate_folder("data_with_field")


if __name__ == "__main__":
    main()
