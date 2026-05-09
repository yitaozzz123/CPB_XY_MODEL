# XY Monte Carlo Simulation

This project implements a two-dimensional XY model using the Metropolis–Hastings Monte Carlo algorithm.

The code includes:
- Monte Carlo simulations,
- vortex and antivortex detection,
- thermalization estimation,
- autocorrelation analysis,
- statistical analysis,
- plot generation.

---

# Requirements

Python 3.10+

```bash
pip install numpy pandas matplotlib tqdm
```

---

# Running the code

## Run the full pipeline

```bash
python main.py
```

This will:
0. Eventually run the simulations and store the data (run_simulations = False)
1. run data checks,
2. run autocorrelation diagnostics,
3. perform the analysis,
4. generate plots.

---

## Run simulations only

```bash
python main_simulations.py
```

---

## Run analysis only

```bash
python main_analysis.py
```

---

# Main parameters

Simulation parameters are defined in `experiments.py`.

Important parameters include:
- temperature range,
- lattice size,
- external field strength,
- number of Monte Carlo sweeps.

Examples:

```python
particles = [10, 20, 50]
```

```python
external_fields = [0.25, 0.5, 1.0, 2.0]
```

The number of sweeps is controlled by:

```python
sweeps_for_configuration()
```

---

# Output folders

## Simulation data

- `data/`
- `data_with_field/`

Simulation data are stored as `.npz` files.

---

## Simulation plots

- `results/`
- `results_with_field/`

Generated plots include:
- lattice configurations,
- magnetization,
- energy,
- vortex density,
- vortex counts.

---

## Analysis output

- `analysis_no_field/`
- `analysis_with_field/`

These folders contain:
- CSV summaries,
- analysis plots,
- KT fit plots,
- power-law plots.

---

# Additional scripts

## Check saved data

```bash
python extra/check_npz.py
```

Checks whether all saved simulation files contain the required observables.

---

## Run autocorrelation diagnostics

```bash
python extra/tau_diagnostic.py
```

Detects simulations with problematic autocorrelation times or block sizes.

---

# Notes

- Periodic boundary conditions are used.
- Natural units are used with:
  - `k_B = 1`
  - `J = 1`
- The code is formatted using `black`.