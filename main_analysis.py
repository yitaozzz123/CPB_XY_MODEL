"""Entry point for analysing XY Monte Carlo simulation data."""

import pandas as pd
from cross_analysis import (
    analyze_data_folder,
    make_field_temperature_comparison_plots,
    make_standard_plots,
    save_analysis_summary,
    plot_vs_temperature,
)


def analyse_no_field_data() -> None:
    """Analyse simulations without external field and save standard plots."""
    no_field_results = analyze_data_folder("data")

    save_analysis_summary(
        no_field_results,
        "analysis_no_field/summary.csv",
    )

    make_standard_plots(
        "analysis_no_field/summary.csv",
        output_folder="analysis_no_field/plots",
        make_temperature_plots=True,
        make_field_plots=False,
    )


def analyse_field_data() -> None:
    """Analyse simulations with external field and save standard plots."""
    field_results = analyze_data_folder("data_with_field")

    save_analysis_summary(
        field_results,
        "analysis_with_field/summary.csv",
    )

    make_standard_plots(
        "analysis_with_field/summary.csv",
        output_folder="analysis_with_field/plots",
        make_temperature_plots=False,
        make_field_plots=True,
    )

    summary = pd.read_csv("analysis_no_field/summary.csv")

    plot_vs_temperature(
        summary,
        observable="tau",
        error=None,
        output_folder="analysis_no_field/plots",
        logy=True,
    )
    make_field_temperature_comparison_plots(
        "analysis_with_field/summary.csv",
        output_folder="analysis_with_field/temperature_comparison_plots",
        lattice_size=20,
    )


def main() -> None:
    """Run all analysis steps."""
    analyse_no_field_data()
    analyse_field_data()


if __name__ == "__main__":
    main()
