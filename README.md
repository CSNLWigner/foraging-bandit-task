# Foraging-like computations in the frontal cortex

Code to reproduce the figures of:

> Ungvárszki Z., Goussi-Denjean C., Székely A., Orbán G., di Volo M., Procyk E. *Neural dynamics reveal foraging-like computations in the frontal cortex* (2026).

Two macaque monkeys performed a three-armed bandit task while neural populations were recorded in the midcingulate cortex (MCC) and in the dorsal and ventral lateral prefrontal cortex (dLPFC, vLPFC). Foraging and reinforcement learning (RL) models were fitted to the behavior, and the value derived from them was tracked in the population activity.

## Installation
The functions that compute the analyses of the paper are part of the `popy` package. The package is named after our monkey Popeye (pronounced as « po-pie »).

To install the package in editable mode, run the following commands.


```bash
conda env create -f environment.yml
conda activate popy
pip install -e .
```

## Data

The data are shared by the lead contact upon request (see the paper). Set the data folder in [popy/config.py](popy/config.py):

```python
DATA_PATH = '/path/to/data'
```

or set the `POPY_DATA_PATH` environment variable. By default, the code looks for a `data/` folder in this repository.

The data folder should contain:

| File | Used for |
|---|---|
| `behavior/behavior_kapo.pkl` | behavior of both monkeys |
| `behavior/recordings_summary.csv` | session metadata (recorded areas, LPFC subregion, recording sites) |
| `neural_data/<monkey>_<session>_spikes.nc` | spike trains of each session |
| `recording_sites/sulci_ka.png`, `recording_sites/sulci_po.png` | sulci drawn on the recording grids |

## Results

The analysis scripts (`run_*.py`) save their results in `RESULTS_PATH` (default: `results/` in this repository, or set the `POPY_RESULTS_PATH` environment variable), where the notebooks read them from. Some of the analyses take hours or days, so the precomputed results are shared together with the data: copy them into `results/` to run the notebooks directly. Each notebook says which script recomputes its results.

The composite sessions (all units pooled, used for Fig. 4a-b and Fig. 6) are not shared: build them with `notebooks/fig4/run_build_composite_session.py`. Their trials are sampled randomly, so the rebuilt composite sessions, and the results computed from them, differ slightly from those of the paper.

## Folder structure

```
popy/                  helper package (data loading, behavioral models, analysis and plotting functions)
  config.py            data, results and figure paths, colors, fitted model parameters
notebooks/
  fig1/                Fig. 1: behavior and model-derived values
  fig2/                Fig. 2: model comparison
  fig3/                Fig. 3: single-unit and population coding of task variables
  fig4/                Fig. 4: neural value and its trial-by-trial update
  fig5/                Fig. 5: neural value within and across targets (foraging vs. RL)
  fig6/                Fig. 6: value updating in a principal component of MCC
results/               results of the analysis scripts (precomputed results are shared with the data)
figs/                  figure panels saved by the notebooks
source_data/           source data of the figures (Source Data.xlsx, and the same tables as CSV files in csv/)
```

Each figure folder contains the notebooks that create the figure panels, the analysis scripts (`run_*.py`) that compute the results used by the notebooks, and helper functions (`helpers*.py`).

## How to run

Run the analysis scripts from their folder (or use the precomputed results), then run the notebooks. `bash run_all.sh` runs all analysis scripts one after the other (about two weeks; logs in `logs/`; `bash run_all.sh 5 6` runs only some of them, see the step numbers in the script). Most scripts use all but a few CPU cores; the scripts that process the spike trains load one session per worker (a few GB each), so on machines with less memory limit the number of workers with the `POPY_N_CPUS` environment variable (e.g. `POPY_N_CPUS=6`).

Run times are approximate (from the original runs and from runs on a 24-core server); they scale with the number of workers.

| Figure | Script to run (in the figure's folder) | Run time | Notebook |
|---|---|---|---|
| Fig. 1d-f | no script needed | -- | `notebooks/fig1/fig1_behavior.ipynb` |
| Fig. 1h-j | no script needed | -- | `notebooks/fig1/fig1_model_values.ipynb` |
| Fig. 2a | `python run_model_optimizing.py` | ~4 h | `notebooks/fig2/fig2a_model_optimizing.ipynb` |
| Fig. 2b | `python run_model_fitting.py` | ~18 h | `notebooks/fig2/fig2b_model_fitting.ipynb` |
| Fig. 2c | `python run_model_fitting.py` (same as Fig. 2b) | ~18 h | `notebooks/fig2/fig2cd_parameter_space.ipynb` |
| Fig. 2d | `python run_bootstrapping.py` | ~2.5 days | `notebooks/fig2/fig2cd_parameter_space.ipynb` |
| Fig. 2e-h | `python run_model_fitting.py --no-cv` | ~2 h | `notebooks/fig2/fig2eh_simulated_behavior.ipynb` |
| Fig. 3b | no script needed | -- | `notebooks/fig3/fig3b_example_units.ipynb` |
| Fig. 3c-e | `python run_glm.py` | ~1.5 days | `notebooks/fig3/fig3cde_single_unit_glm.ipynb` |
| Fig. 3f | `python run_decoding.py fb_target` and `python run_decoding.py stay_value` | ~3 days each | `notebooks/fig3/fig3f_population_decoding.ipynb` |
| Fig. 4a-b | `python run_build_composite_session.py meta_rates_history` (required, not shared) | ~3 h | `notebooks/fig4/fig4ab_composite_session.ipynb` |
| Fig. 4c-f | `python run_neural_value_all_sessions.py` | ~1 h | `notebooks/fig4/fig4cdef_neural_value.ipynb` |
| Fig. 4g | `python run_decoding.py multiple_alphas_ridge` (in `notebooks/fig3/`) | ~3 days | `notebooks/fig4/fig4g_learning_rate.ipynb` |
| Fig. 5c-e | `python run_neural_value_across_target.py before_fb` | ~5 h | `notebooks/fig5/fig5cde_across_target.ipynb` |
| Fig. 6 | `python run_build_composite_session.py meta_rates_value_4x20` (in `notebooks/fig4/`; required, not shared) | ~3 h | `notebooks/fig6/fig6_pca.ipynb` |
| Supp. Fig. 1 | no script needed | -- | `notebooks/fig1/figS1_delta_rule.ipynb` |
| Supp. Fig. 3 | `python run_model_recovery.py` | ~13 h | `notebooks/fig2/figS3_model_recovery.ipynb` |
| Supp. Fig. 4 | `python run_model_recovery.py` (same as Supp. Fig. 3) | ~13 h | `notebooks/fig2/figS4_parameter_recovery.ipynb` |
| Supp. Fig. 5 | `python run_model_fitting_first_trials.py 10` and `python run_model_fitting_first_trials.py 15` (all trials: `python run_model_fitting.py`, as Fig. 2b) | ~4 h each | `notebooks/fig2/figS5_first_trials_fitting.ipynb` |
| Supp. Fig. 6 | `python run_decoding.py alternative_model_values` | ~1.5 days | `notebooks/fig3/figS6_q_value_decoding.ipynb` |
| Supp. Fig. 7 | `python run_decoding.py fb_target` (a-b, as Fig. 3f) and `python run_decoding.py stay_value_current_trial` (c) | ~3 days and ~3 h | `notebooks/fig3/figS7_decoding_per_monkey.ipynb` |
| Supp. Fig. 8 | `python run_build_composite_session.py meta_rates_history` (required, not shared) | ~3 h | `notebooks/fig4/figS8_lpfc_dynamics.ipynb` |
| Supp. Fig. 9 | `python run_neural_value_all_sessions.py` | ~1 h | `notebooks/fig4/figS9_per_monkey_neural_value.ipynb` |
| Supp. Fig. 10 | `python run_neural_value_all_sessions.py` | ~1 h | `notebooks/fig4/figS10_delta_rule_glm.ipynb` |
| Supp. Fig. 11 | `python run_decoding.py multiple_alphas_ridge` (in `notebooks/fig3/`) | ~3 days | `notebooks/fig4/figS11_best_learning_rate.ipynb` |
| Supp. Fig. 12 | `python run_decoding.py multiple_alphas_ridge` (in `notebooks/fig3/`) | ~3 days | `notebooks/fig4/figS12_learning_rate_vlpfc.ipynb` |
| Supp. Fig. 13 | `python run_neural_value_across_target.py before_fb` | ~5 h | `notebooks/fig5/figS13_per_monkey_hypotheses.ipynb` |
| Supp. Fig. 14 | `python run_neural_value_across_target.py` (all six time windows) | ~1 day | `notebooks/fig5/figS14_time_windows.ipynb` |

The model parameters fitted to the monkeys (from `run_model_fitting.py --no-cv`) are stored in [popy/config.py](popy/config.py) (`MODEL_PARAMS` for the foraging model, `MODEL_PARAMS_sRL` for the inferential RL model) and are used to compute the model-derived values.

Panels not listed (Fig. 1a-c, g, Fig. 3a, Fig. 5a-b, Supp. Fig. 2 and Supp. Fig. 14a) are schematics or illustrations. The supplementary-figure notebooks are in the folder of the related main figure (`figS<N>_*.ipynb`).

## Source data

The notebooks also export the data plotted in each figure panel (with [popy/source_data.py](popy/source_data.py)): `source_data/Source Data.xlsx` has one sheet per panel (e.g. `Figure 2a`) with a short description and the data table(s); `source_data/csv/<panel>/` has the same tables as CSV files.

## Acknowledgements

The code documentation (docstrings, comments and this README) was written with the help of Claude Opus 5.5 (`claude-opus-5-5`, Anthropic), and reviewed by the author.
