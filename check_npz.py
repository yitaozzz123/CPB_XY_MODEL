"""Check whether stored simulation files contain vortex observables."""

from pathlib import Path

import numpy as np

REQUIRED_KEYS = {
    "vortex_density",
    "n_vortices",
    "n_antivortices",
}


def check_npz_file(filename: str | Path, required_keys: set[str]) -> None:
    """Print whether one NumPy archive contains all required keys."""
    with np.load(filename, allow_pickle=True) as data:
        available_keys = set(data.files)
        missing_keys = required_keys - available_keys

        if missing_keys:
            print(f"[MISSING] {filename}")
            print(f"  missing: {sorted(missing_keys)}")
            print(f"  available: {sorted(available_keys)}")
            return

        print(f"[OK]      {filename}")


def check_data_folder(
    data_folder: str | Path = "data",
    required_keys: set[str] | None = None,
) -> None:
    """Check all simulation archives in a folder."""
    if required_keys is None:
        required_keys = REQUIRED_KEYS

    for filename in sorted(Path(data_folder).glob("*_data.npz")):
        check_npz_file(filename, required_keys)


def main() -> None:
    """Check all stored no-field simulation files."""
    check_data_folder("data")


if __name__ == "__main__":
    main()
