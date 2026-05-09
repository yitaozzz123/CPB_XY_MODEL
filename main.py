"""Top-level entry point for simulations, diagnostics, and analysis."""

import subprocess
import sys
from pathlib import Path
import os


def run_script(script_name: str) -> None:
    """Run a Python script and raise an error if it fails."""
    project_root = Path(__file__).resolve().parent

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [sys.executable, script_name],
        check=False,
        cwd=project_root,
        env=environment,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error while running {script_name}")


def main() -> None:
    """Run the full simulation, diagnostic, and analysis pipeline."""

    run_simulations = True

    if run_simulations:
        print("=== Running simulations ===")
        run_script("main_simulations.py")

    print("=== Checking saved data ===")
    run_script("extra/check_npz.py")

    print("=== Running autocorrelation diagnostics ===")
    run_script("extra/tau_diagnostic.py")

    print("=== Running analysis ===")
    run_script("main_analysis.py")

    print("=== All done ===")


if __name__ == "__main__":
    main()
