"""
Helper functions for the Fig. 4 notebooks.
"""
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from popy.io_tools import load_metadata
from popy.config import RESULTS_PATH
from helpers_analysis import simple_glm_weights, get_cpd_all


##% data loading (Fig. 4a-b)
def load_mega_custom(monkey=None, area=None, subregion=None, downsample=True):
    # load data (meta session)
    floc = os.path.join(RESULTS_PATH, 'composite_sessions', 'meta_rates_history')
    neural_dataset =  xr.open_dataset(floc + '/meta_rates.nc')

    if area is not None:
        neural_dataset = neural_dataset.sel(unit=neural_dataset.area == area)
    if subregion is not None:
        neural_dataset = neural_dataset.sel(unit=neural_dataset.subregion == subregion)

    if monkey is not None:
        neural_dataset = neural_dataset.sel(unit=neural_dataset.monkey == monkey)

    # use only time points of every .1 s
    if downsample:
        time_vector = neural_dataset.time.values
        time_vector = np.unique(np.round(time_vector, 1))  # round to 1 decimal place to avoid floating point issues
        neural_dataset = neural_dataset.sel(time=time_vector, method='nearest')

    # add coordinates (should be added during the dataset creation)
    neural_dataset = neural_dataset.assign_coords(R_1=('trial_id', [0 if hist in [0, 1, 2, 3] else 1 for hist in neural_dataset.history_of_feedback.values]))
    neural_dataset = neural_dataset.assign_coords(R_2=('trial_id', [0 if hist in [0, 1, 4, 5] else 1 for hist in neural_dataset.history_of_feedback.values]))
    neural_dataset = neural_dataset.assign_coords(R_3=('trial_id', [0 if hist in [0, 2, 4, 6] else 1 for hist in neural_dataset.history_of_feedback.values]))
    neural_dataset = neural_dataset.assign_coords(fb_sequence=('trial_id', neural_dataset.history_of_feedback.values))
    neural_dataset = neural_dataset.assign_coords(fb_sequence_m1=('trial_id', neural_dataset.history_of_feedback.values%4))
    neural_dataset = neural_dataset.assign_coords(time_m1=('time', neural_dataset.time.values + 7.5))

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


# Fig. 4c-e
def prepare_short_history_data(behav_new, time_interval, n_perms):
    """Session averages, GLM weights and CPDs (with permutation p-values) shared by both plotting modes."""
    # rename columns for plotting: V_T_... to V_t, dV_... to dV
    behav_new['V_t'] = behav_new[f'V_t_{time_interval[0]}_{time_interval[1]}']
    behav_new['V_t_p1'] = behav_new[f'V_t_p1_{time_interval[0]}_{time_interval[1]}']
    behav_new['dV'] = behav_new[f'dV_{time_interval[0]}_{time_interval[1]}']

    # for each monkey, session, subregion, fb_sequence, take a mean of these columns
    behav_new_sess_avg_sequence = behav_new.copy().groupby(['monkey', 'session', 'subregion', 'fb_sequence'])['V_t'].mean().reset_index()

    behav_new_sess_avg_fb = []
    for (monkey, session, subregion), group in behav_new.groupby(['monkey', 'session', 'subregion']):
        U, pval = stats.mannwhitneyu(
            group.loc[group['feedback'] == 0, 'dV'],
            group.loc[group['feedback'] == 1, 'dV'],
            alternative='two-sided'
        )
        for fb in group['feedback'].unique():
            behav_new_sess_avg_fb.append({
                'monkey': monkey,
                'session': session,
                'subregion': subregion,
                'feedback': fb,
                'V_t': group.loc[group['feedback'] == fb, 'V_t'].mean(),
                'V_t_p1': group.loc[group['feedback'] == fb, 'V_t_p1'].mean(),
                'dV': group.loc[group['feedback'] == fb, 'dV'].mean(),
                'U': U,
                'signif': pval < 0.05
            })
    behav_new_sess_avg_fb = pd.DataFrame(behav_new_sess_avg_fb)

    return {
        'behav': behav_new,
        'sess_avg_sequence': behav_new_sess_avg_sequence,
        'sess_avg_fb': behav_new_sess_avg_fb,
        # calculate simple GLM weights for each session, subregion, monkey
        'weights': simple_glm_weights(behav_new),
        # Main analysis with joblib parallelization
        'cpd': get_cpd_all(behav_new, n_perms),
        'n_perms': n_perms,
    }


##% data tools (Fig. 4g)
def load_decoding_res_new(folder='basics'):
    xr_res = xr.open_dataset(os.path.join(RESULTS_PATH, 'population_decoding', folder, 'scores.nc'))
    xr_res.load()

    if 'group_target' in xr_res.dims:
        if xr_res.group_target == 'None':
            xr_res = xr_res.mean('group_target')
        else:
            raise ValueError("group_target was used so the plotting method cant handle this.")


    # remove nans along 'session' dimension
    #xr_res = xr_res.dropna('session')
    
    return xr_res


def mask_N_consecutives(mask_xr, N=4):
    """
    Takes a mask (True/False) and sets to False any True value that is not part of N consecutive True values.
    
    Parameters
    ----------
    data : Xarray
        3D Xarray with dimensions (session, time, area)
    N : int
        Number of consecutive non-significant bins to mask
        
    Returns
    -------
    data_masked : Xarray
        3D Xarray with dimensions (session, time, area) with N consecutive non-significant bins masked.
    """

    # find where N consecutive values are False
    mask = np.empty(mask_xr.shape)
    for sess_id in range(mask_xr.shape[0]):
        for area_id in range(mask_xr.shape[2]):
            # get session data
            session_temp = mask_xr[sess_id, :, area_id].values

            consecutive_count = 0
            candidate_ids = []
            unit_signif_ids = []
            for i, if_signif_bin in enumerate(session_temp):
                # if not False, add to count
                if if_signif_bin:
                    consecutive_count += 1
                    candidate_ids.append(i)  # add to candidate indices (may or may not be part of N consecutive bins)
                    if consecutive_count >= N:
                        unit_signif_ids += candidate_ids  # add to list of significant indices
                        candidate_ids = []  # reset candidate ids
                else:  # reset
                    consecutive_count = 0
                    candidate_ids = []
            # create mask for this session (1 if significant, 0 if not) 
            unit_signif = np.zeros_like(session_temp)
            unit_signif[unit_signif_ids] = 1

            mask[sess_id, :, area_id] = unit_signif

    #data_masked = data.where(mask == 1)
    mask = mask.astype(bool)
    mask_xr.data = mask
    return mask_xr


def add_subareas(res):
    metadata = load_metadata()

    subregions = []
    for sess_temp in res.session.values:
        res_temp = res.sel(session=sess_temp)
        # get session metadata
        monkey, session = res_temp.monkey, sess_temp.split('_')[1]
        lpfc_subregion = metadata.loc[(metadata['monkey'] == monkey) & (metadata['session'] == session), 'LPFC_subregion'].values[0]
        subregions.append(lpfc_subregion)

    # add new coordinate along (session, area) dimension
    res = res.assign_coords(lpfc_subregion=('session', subregions))

    return res


def add_if_signif(res, plim=.05, N_consecutives=4):
    res_teagrets = []
    for j, target in enumerate(res.target.values):
        # mask significants, convert to 0 or 1 instead of p values
        res_temp = res.sel(target=target)
        res_temp_signifs = mask_N_consecutives(res_temp.pvals <= plim, N=N_consecutives)  # mask N consecutive non-significant bins
        res_temp['if_significant'] = res_temp_signifs  # add as new dataarray to the results

        res_teagrets.append(res_temp)

    res = xr.concat(res_teagrets, dim='target')
    return res  

def remove_irrelevant_sessions(res):
    sessions_good = []
    for session in res.session.data:
        for area in res.area.data:
            res_temp = res.sel(session=session, area=area)

            if np.any(res_temp.if_significant.sel(time=slice(2, 3.5)).sum('time') > 0) and np.any(res_temp.scores.sel(time=slice(2, 3.5)).sum('time') > 0.1):
                sessions_good.append((session, area))
            else:
                res.loc[dict(session=session, area=area)] = np.nan
                
    print(f'kept: {len(sessions_good)} / {len(res.session.data)*len(res.area.data)}')
    return res
