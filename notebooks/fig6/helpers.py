"""
Helper functions for the Fig. 6 notebook.
"""
import os

import numpy as np
import xarray as xr
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm
import statsmodels.formula.api as smf

from popy.config import RESULTS_PATH

#%% data tools
def load_data_custom():
    # load data (meta session)
    floc = os.path.join(RESULTS_PATH, 'composite_sessions', 'meta_rates_value_4x20')
    neural_dataset =  xr.open_dataset(floc + '/meta_rates.nc')

    # normalize firing rates (z-score) and remove units with low firing rates
    threshold = 1  # Hz
    units_to_remove = []
    normalized_data = np.zeros_like(neural_dataset.firing_rates.data)  # Create a copy to avoid modifying the original data
    for i, unit in enumerate(neural_dataset.unit.values):  # Loop through each neuron
        # Get all data for this neuron across trials and timepoints
        unit_data = neural_dataset.firing_rates.sel(unit=unit).values
        
        # Calculate mean and std across all values for this neuron
        mean_val = np.mean(unit_data)
        std_val = np.std(unit_data)
        
        # Z-score normalize and store in the output array
        if mean_val < threshold:
            units_to_remove.append(unit)
        elif std_val == 0:
            raise ValueError(f"Standard deviation is zero for unit {unit}. Not possible.")
        else:
            z_scored_data = (unit_data - mean_val) / std_val        
            normalized_data[:, i, :] = z_scored_data

    #neural_dataset.firing_rates.values = normalized_data
    neural_dataset['firing_rates'] = (neural_dataset.firing_rates.dims, normalized_data)

    neural_dataset = neural_dataset.sel(unit=~neural_dataset.unit.isin(units_to_remove))
    print(f"Removed {len(units_to_remove)} units with mean firing rate < {threshold} Hz")

    return neural_dataset

def calculate_mixed_effect(value, feedback, V_t, dV):
    # Assume your DataFrame has columns: V_t, dV, value, feedback
    # df = pd.DataFrame(...)
    df = pd.DataFrame({"dV": dV, "fb": feedback, "V_t": V_t, "value": value})

    # 1) How much 'value' explains 'V_t' using ANOVA (value has 4 classes)
    model_val = smf.ols("V_t ~ C(value)", data=df).fit()
    anova_val = sm.stats.anova_lm(model_val, typ=2)
    ss_between_val = anova_val.loc["C(value)", "sum_sq"]
    ss_total_val = anova_val["sum_sq"].sum()
    eta_sq_value = ss_between_val / ss_total_val
    pval_value = anova_val.loc["C(value)", "PR(>F)"]

    # 2) How much 'feedback' explains 'dV' using ANOVA
    model_fb = smf.ols("dV ~ C(feedback)", data=df).fit()
    anova_fb = sm.stats.anova_lm(model_fb, typ=2)
    ss_between_fb = anova_fb.loc["C(feedback)", "sum_sq"]
    ss_total_fb = anova_fb["sum_sq"].sum()
    eta_sq_fb = ss_between_fb / ss_total_fb
    pval_fb = anova_fb.loc["C(feedback)", "PR(>F)"]

    return (eta_sq_value, pval_value), (eta_sq_fb, pval_fb)

def fit_transform_data(neural_dataset):
    n_trials, n_neurons, n_time = neural_dataset.firing_rates.data.shape
        
    # Step 1: Reshape to (neurons, trials*time)
    reshaped_data = neural_dataset.firing_rates.data.transpose(1, 0, 2).reshape(n_neurons, -1)

    # Step 2: Apply PCA (scikit-learn expects samples as rows, features as columns)
    pca = PCA()
    pca = pca.fit(reshaped_data.T)  # Shape: (trials*time, components)
    transformed_data = pca.transform(reshaped_data.T)  # Apply PCA transformation

    # Step 3: Reshape back to (trials, components, time)
    transformed_data = transformed_data.T  # Shape: (components, trials*time)
    transformed_data = transformed_data.reshape(n_neurons, n_trials, n_time)
    transformed_data = transformed_data.transpose(1, 0, 2)  # Final shape: (trials, components, time)

    # rewrite the dataset
    pca_dataset = neural_dataset.copy(deep=True)
    pca_dataset = pca_dataset.rename({'unit': 'pca_component'})
    pca_dataset = pca_dataset.drop('channel')
    pca_dataset = pca_dataset.drop('area')
    pca_dataset = pca_dataset.drop('unit_id_original')
    pca_dataset = pca_dataset.drop('monkey')
    pca_dataset = pca_dataset.drop('session')
    # add pca_component dimension
    pca_dataset = pca_dataset.assign_coords(pca_component=('pca_component', np.arange(n_neurons)))
    pca_dataset = pca_dataset.assign_coords(explained_var=('pca_component', pca.explained_variance_ratio_))
    pca_dataset.firing_rates.data = transformed_data

    # create xr dataset for the pca components (components, units)
    pca_components = xr.DataArray(pca.components_, dims=['pca_component', 'unit'], coords=[np.arange(n_neurons), neural_dataset.unit.values])

    return pca_dataset, pca_components

def compute_explained_var(pca_dataset_all, n_comps=10000):
    res = {'ka': {}, 'po': {}}
    for monkey in ['ka', 'po']:
        for area in ['MCC', 'vLPFC', 'dLPFC']:
            pca_dataset = pca_dataset_all[monkey][area]

            n_components = np.min((n_comps, len(pca_dataset.pca_component.values)))

            r2s_value, r2s_feedback = [], []
            pvals_value, pvals_feedback = [], []
            for i in range(n_components):  # loop through PCs and calculate variance
                # prepare data for this PC
                V_t = pca_dataset.firing_rates.sel(pca_component=i, time=slice(1, 3.5)).mean('time').data  # select PC, get one mean value per trial
                V_tp1 = pca_dataset.firing_rates.sel(pca_component=i, time=slice(8.5, 11)).mean('time').data  # select PC, get one mean value per trial
                dV = V_tp1 - V_t  # difference between timepoints

                behavioral_values = pca_dataset.value_function.values  # get behavioral values
                fbs = pca_dataset.feedback.values  # get feedback values

                # new way 2: mixed effect model
                (r_squared_value, pvalue_value), (r_squared_feedback, p_value_feedback) = calculate_mixed_effect(behavioral_values, fbs, V_t, dV)

                r2s_value.append(r_squared_value)
                r2s_feedback.append(r_squared_feedback)
                pvals_value.append(pvalue_value)
                pvals_feedback.append(p_value_feedback)
            r2s_value = np.array(r2s_value)
            pvals_value = np.array(pvals_value)
            r2s_feedback = np.array(r2s_feedback)
            pvals_feedback = np.array(pvals_feedback)

            res[monkey][area] = {
                'r2s_value': r2s_value,
                'pvals_value': pvals_value,
                'r2s_feedback': r2s_feedback,
                'pvals_feedback': pvals_feedback,
            }

    return res
