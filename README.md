# PEMA-Work: simulation code

Personality-EMotion Agent model of workplace allostatic load, OCB-to-CWB erosion and employee turnover.
Code and figure scripts for the manuscript "Reading the warning signs: a cybernetic agent-based model of personality, emotion and allostatic load in employee turnover".

## Requirements
Python >= 3.10, numpy, scipy, scikit-learn, matplotlib. Create the folders `out/` and `fig/` first (they are included, empty).

## Reproducing the results
| Command | Produces |
|---|---|
| `python experiments.py shock` | P1 controlled-shock experiment (2,000 agents) |
| `python experiments.py baseline` | 50 baseline replications (N = 300, T = 1,000 days; seeds 0-49) |
| `python experiments.py scen` | P5 policy scenarios (30 replications each; seeds 5000-5029) |
| `python experiments.py sa` | Sobol sensitivity runs (1,152 runs; N = 150, T = 730) |
| `python analyse_baseline.py` | P1-P4 statistics and Figures 2-3 (`out/baseline_summary.json`) |
| `python analyse_sa_scen.py` | Sobol indices and policy results, Figures 4-5 |
| `python fig_arch.py` | Figure 1 (architecture) |
| `python ablation.py` | Hazard-specification check (Section 6.3), `out/ablation.json` |

`pema_work_abl.py` is `pema_work.py` with one extra option (`sat_w`, `sat_s0`) that lets the exit hazard depend partly on low satisfaction; with the default `sat_w = 0` it is identical to the baseline model.

## Files
- `pema_work.py`: model (Omega matrix, OCC event types, Mamdani rules, dynamics, hazard, policies).
- `experiments.py`, `analyse_*.py`, `fig_arch.py`, `ablation.py`: experiments and figures.
- `figures/`: the figures as used in the manuscript.

Baseline parameter values are listed in `DEFAULTS` in `pema_work.py` and in Table 2 of the manuscript.
