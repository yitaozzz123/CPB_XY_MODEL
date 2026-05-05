from experiments import run_temperature_scan, run_external_field_scan
from cross_analysis import (
    analyze_data_folder,
    save_analysis_summary,
    make_standard_plots,
)


def main():
    run_temperature_scan()
    run_external_field_scan()

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


if __name__ == "__main__":
    main()
