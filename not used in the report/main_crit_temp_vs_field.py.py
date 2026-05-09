from experiments import run_critical_temperature_study_simulations
from analysis_temperature_vs_field import analysis_temp_vs_field


def main():
    run_critical_temperature_study_simulations()
    analysis_temp_vs_field()


if __name__ == "__main__":
    main()
