"""Entry point for running all XY Monte Carlo simulations."""

from experiments import run_external_field_scan, run_temperature_scan


def main() -> None:
    """Run all simulation scans."""
    run_temperature_scan()
    run_external_field_scan()


if __name__ == "__main__":
    main()
