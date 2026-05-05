from cross_analysis import (
    analyze_data_folder,
    save_analysis_summary,
    make_standard_plots,
    make_temperature_power_law_plots,
    make_field_temperature_comparison_plots,
    plot_KT_fit,
)

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

import pandas as pd

summary = pd.read_csv("analysis_no_field/summary.csv")

observables = [
    "magnetic_susceptibility_per_spin",
    "tau",
    "mean_vortex_density",
]

for obs in observables:
    plot_KT_fit(
        summary,
        observable=obs,
        Tc=0.88,
        output_folder="analysis_no_field/KT_plots",
    )


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
    n_particles_1d=20,
)
