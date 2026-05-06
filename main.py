"""Top-level entry point for simulations, diagnostics, and analysis."""

import subprocess
import sys


def run_script(script_name: str) -> None:
    """Run a Python script and raise an error if it fails."""
    result = subprocess.run(
        [sys.executable, script_name],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error while running {script_name}")


def main() -> None:
    """Run the full simulation, diagnostic, and analysis pipeline."""

    run_simulations = False

    if run_simulations:
        print("=== Running simulations ===")
        run_script("main_simulations.py")

    print("=== Checking saved data ===")
    run_script("check_npz.py")

    print("=== Running autocorrelation diagnostics ===")
    run_script("tau_diagnostic.py")

    print("=== Running analysis ===")
    run_script("main_analysis.py")

    print("=== All done ===")


if __name__ == "__main__":
    main()
