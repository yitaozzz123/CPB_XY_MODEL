"""Entry point for analysing XY Monte Carlo simulation data."""

import pandas as pd

from cross_analysis import (
    analyze_data_folder,
    make_field_temperature_comparison_plots,
    make_standard_plots,
    make_temperature_power_law_plots,
    plot_KT_fit,
    save_analysis_summary,
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

    make_temperature_power_law_plots(
        "analysis_no_field/summary.csv",
        output_folder="analysis_no_field/power_law_plots",
        fit_summary_file="analysis_no_field/power_law_fit_summary.csv",
    )

    summary = pd.read_csv("analysis_no_field/summary.csv")

    kt_observables = [
        "magnetic_susceptibility_per_spin",
        "tau",
        "mean_vortex_density",
    ]

    for observable in kt_observables:
        plot_KT_fit(
            summary,
            observable=observable,
            critical_temperature=0.88,
            output_folder="analysis_no_field/KT_plots",
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
