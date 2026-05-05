import subprocess
import sys


def run_script(script_name):
    result = subprocess.run([sys.executable, script_name])

    if result.returncode != 0:
        raise RuntimeError(f"Error on {script_name}")


def main():
    print("=== Running simulations ===")
    run_script("main_simulations.py")

    print("=== Running analysis ===")
    run_script("main_analysis.py")

    print("=== All done ===")


if __name__ == "__main__":
    main()
