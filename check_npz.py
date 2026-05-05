from pathlib import Path
import numpy as np

required_keys = {
    "vortex_density",
    "n_vortices",
    "n_antivortices",
}

data_folder = Path("data")

for filename in sorted(data_folder.glob("*_data.npz")):
    with np.load(filename, allow_pickle=True) as data:
        keys = set(data.files)
        missing = required_keys - keys

        if missing:
            print(f"[MANCANO] {filename}")
            print(f"  mancanti: {sorted(missing)}")
            print(f"  presenti: {sorted(keys)}")
        else:
            print(f"[OK]      {filename}")
