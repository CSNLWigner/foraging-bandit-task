"""
Functions to transform the session data dataframe. The dataframe is the already processed pandas df, not the raw data!
Exception: the last section contains the raw data preprocessing of monkey Yuri (csv files -> session data dataframe).
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit

import popy.config as cfg
from popy.simulation_tools import *

def add_trial_in_block(behav):
    """
    Adds a 'trial_id_in_block' column to the behav DataFrame, which represents the trial index within each block.

    Parameters:
    behav (pandas.DataFrame): 
        The input DataFrame containing the session data.

    Returns:
    pandas.DataFrame: 
        The modified DataFrame with the 'trial_id_in_block' column added.
    """
    
    # if already has the property, just return as it is
    if 'trial_id_in_block' in behav.columns:
        return behav
    # group by monkey, session, block
    for info, subdf in behav.groupby(['monkey', 'session', 'block_id']):
        # write back to the original dataframe
        behav.loc[subdf.index, 'trial_id_in_block'] = np.arange(len(subdf))

    # type is int
    behav['trial_id_in_block'] = behav['trial_id_in_block'].astype(int)

    return behav

def drop_time_fields(behav):
    """
    Drop the time fields from the session data.
    """

    # drop columns ending with '_time'
    behav = behav.drop([col for col in behav.columns if col.endswith('_time')], axis=1)

    return behav

def convert_column_format(behav, original='simulation'):
    """
    Converts column names and values from behavioral to modeling format (because unfortunately different naming and coding were used).

    Parameters:
    ------------
    behav: pd.DataFrame
        The session data dataframe.
    original: str
        The original format of the data. Can be 'behavior' or 'simulation'.

    Returns:
    ------------
    behav: pd.DataFrame
        The session data dataframe with the column names and values converted to the modeling format
    """
    if original == 'behavior':
        # rename target to action, also convert values from [1,2,3] to [0,1,2]
        target = behav['target'].values
        target[target==1] = 0
        target[target==2] = 1
        target[target==3] = 2
        behav['target'] = target
        # rename to action
        behav = behav.rename(columns={'target': 'action'})

        # rename feedback to reward
        behav = behav.rename(columns={'feedback': 'reward'})

        # change best target to best arm
        best_target = behav['best_target'].values
        best_target[best_target==1] = 0
        best_target[best_target==2] = 1
        best_target[best_target==3] = 2
        behav['best_target'] = best_target
        behav = behav.rename(columns={'best_target': 'best_arm'})
    
    elif original == 'simulation':
        # rename action to target, also convert values from [0,1,2] to [1,2,3]
        actions = behav['action'].values
        actions[actions==2] = 3
        actions[actions==1] = 2
        actions[actions==0] = 1
        behav['action'] = actions
        behav = behav.rename(columns={'action': 'target'})

        # rename reward to feedback
        behav = behav.rename(columns={'reward': 'feedback'})

        # change best arm to best target
        best_arm = behav['best_arm'].values
        best_arm[best_arm==2] = 3
        best_arm[best_arm==1] = 2
        best_arm[best_arm==0] = 1
        behav['best_arm'] = best_arm
        behav = behav.rename(columns={'best_arm': 'best_target'})

    else:
        raise ValueError("Invalid original format.")
        
    return behav

def add_history_of_feedback(
    behav_original,
    num_trials=8,
    one_column=True,
    coding=(0, 1),
    add_history_of_targets=True
):
    """
    Add feedback/target history to a dataframe grouped by ['monkey', 'session'].

    If one_column is False:
        - Adds separate columns R_1 ... R_num_trials using 'coding' for 0/1 feedback.

    If one_column is True:
        - Computes a single column 'history_of_feedback' that encodes the last
          num_trials feedbacks as a binary number where R_1 has the highest weight,
          then R_2, ..., e.g. [R_1, R_2, R_3] = [1, 0, 1] -> 5.

    Parameters
    ----------
    behav_original : pandas.DataFrame
        Must contain at least columns: 'feedback', 'target', 'monkey', 'session'.
        'feedback' is expected to be 0/1 (non-rewarded/rewarded).

    num_trials : int
        Number of past trials to include.

    one_column : bool
        See above.

    coding : tuple
        (code_for_nonreward, code_for_reward) used only when one_column=False.

    add_history_of_targets : bool
        Whether to also add T_1 ... T_num_trials (shifted 'target').

    Returns
    -------
    pandas.DataFrame
        Input dataframe with added columns.
    """
    behav = behav_original.copy()

    # Clean up any pre-existing columns we may recreate
    if 'history_of_feedback' in behav.columns:
        behav = behav.drop(columns=['history_of_feedback'])
    for j in range(1, num_trials + 1):
        behav = behav.drop(columns=[f'R_{j}'], errors='ignore')
        behav = behav.drop(columns=[f'T_{j}'], errors='ignore')

    subdfs = []
    group_cols = ['monkey', 'session']
    if not all(col in behav.columns for col in group_cols):
        raise KeyError(f"Missing required grouping columns {group_cols}")

    # We will use raw 0/1 feedback for binary encoding (one_column=True),
    # and use 'coding' only when creating separate R_j columns (one_column=False).
    for _, subdf in behav.groupby(group_cols, sort=False):
        # Shift feedback/targets within each (monkey, session)
        if one_column:
            # For binary number, we need pure 0/1 history
            fb_shift_cols = []
            for j in range(1, num_trials + 1):
                col = f'_Rbin_{j}'
                subdf[col] = subdf['feedback'].shift(j).astype('float')  # 0/1 or NaN
                fb_shift_cols.append(col)
        else:
            # Separate columns coded with 'coding'
            for j in range(1, num_trials + 1):
                subdf[f'R_{j}'] = subdf['feedback'].shift(j).replace({0: coding[0], 1: coding[1]})

        if add_history_of_targets:
            for j in range(1, num_trials + 1):
                subdf[f'T_{j}'] = subdf['target'].shift(j)

        subdfs.append(subdf)

    behav = pd.concat(subdfs, ignore_index=True)

    if one_column:
        # Vectorized binary encoding: R_1 is MSB, then R_2, ..., R_num_trials
        fb_shift_cols = [f'_Rbin_{j}' for j in range(1, num_trials + 1)]
        # If any NaN in a row -> result NaN
        fb_mat = behav[fb_shift_cols].to_numpy(dtype=float)
        has_nan = np.isnan(fb_mat).any(axis=1)

        # Weights: [2^(num_trials-1), ..., 2^0]
        weights = 2 ** np.arange(num_trials - 1, -1, -1, dtype=float)

        # For rows without NaN, compute the dot product; else NaN
        binary_vals = np.where(
            has_nan,
            np.nan,
            fb_mat.dot(weights)
        )

        behav['history_of_feedback'] = binary_vals.astype(float)

        # Drop temp columns
        behav = behav.drop(columns=fb_shift_cols)

    return behav

def add_foraging_value(behav_original, alpha=None, digitize=False, n_classes=4, reset_on_switch=True, add_proba=False, add_PE=False):
    """
    VALUE CORRESPONDING TO TRIAL t IS THE ONE KNOWN BEFORE THE FEEDBACK OF TRIAL t IS RECEIVED!

    in case of interrupted trials (NaN actions), the value function is untouched.

    Model-based value function calculation. 
    
    The value function is calculated based on the Shift-value model with reset.
    """

    behavior = behav_original.copy()
    behavior.reset_index(drop=True, inplace=True)  # reset index 

    for monkey, behav in behavior.groupby('monkey'):
        # init an agent
        agent_class = ForagingAgent
        fixed_params = {'reset_on_switch': reset_on_switch}
        params = cfg.MODEL_PARAMS[monkey].copy()  # get the parameters for the monkey (copy, so that a custom alpha does not change the config)
        if alpha is not None:
            params['alpha'] = alpha

        agent = agent_class(**params, **fixed_params)

        stay_values = np.zeros(len(behav))
        stay_probas = np.zeros(len(behav))
        pes = np.zeros(len(behav))
        session_previous = 'none'
        for i, (_, row) in enumerate(behav.iterrows()):
            # If we are in a new session, reset the agent
            session_curr = row["session"]
            if session_curr != session_previous:
                agent.reset()
                session_previous = session_curr

            # Get latent value function (before update)
            stay_values[i] = agent.V
            stay_probas[i] = agent._get_stay_proba()

            # Get the action and reward taken by the agent
            action, reward = row["target"], row["feedback"]

            # Skip updating for NaN actions (interrupted trials)
            if np.isnan(action):
                #stay_values[i] = np.nan
                #stay_probas[i] = np.nan
                continue

            # update the agent
            agent.update_values(int(action) - 1, int(reward))
            # save pe used for value updating
            pes[i] = agent.pe

        behav.loc[:, 'stay_value'] = stay_values
        if add_proba:
            behav.loc[:, 'stay_proba'] = stay_probas
        if add_PE:
            behav.loc[:, 'PE'] = pes

        # write back the q_values to the original dataframe, matching the indices
        behavior.loc[behav.index, 'stay_value'] = behav.loc[behav.index, 'stay_value'].values
        behavior['V0'] = params['V0']
        if add_proba:
            behavior.loc[behav.index, 'stay_proba'] = behav.loc[behav.index, 'stay_proba'].values
        if add_PE:
            behavior.loc[behav.index, 'PE_foraging'] = behav.loc[behav.index, 'PE'].values

    if digitize:
        # Create bin edges (from min to max value, or 0-1 if specified)
        min_V, max_V = 0, 1
        bin_edges = np.linspace(min_V, max_V, n_classes+1)
                
        # Perform the binning
        behavior['stay_value'] = np.digitize(
            behavior['stay_value'], 
            bins=bin_edges, 
            right=False) - 1
        
        min_proba_stay, max_proba_stay = agent._get_min_max_stay_proba()
        bins_edges = np.linspace(min_proba_stay, max_proba_stay, n_classes+1)
        
        if add_proba:
            behavior['stay_proba'] = np.digitize(
                behavior['stay_proba'], 
                bins=bins_edges, 
                right=False) - 1

    return behavior

def add_RL_values(behav_original, digitize=False, n_classes=4, structure_aware=True, add_proba=False, add_PE=False):
    """
    VALUE CORRESPONDING TO TRIAL t IS THE ONE KNOWN BEFORE THE FEEDBACK OF TRIAL t IS RECEIVED!

    Model-based value function calculation. 
    
    The value function is calculated based on the Shift-value model with reset.
    """

    behavior = behav_original.copy()
    behavior.reset_index(drop=True, inplace=True)  # reset index 

    for monkey, behav in behavior.groupby('monkey'):
        # init an agent
        agent_class = QLearner
        fixed_params = {'structure_aware': structure_aware}
        params = cfg.MODEL_PARAMS_sRL[monkey]  # get the parameters for the monkey

        agent = agent_class(**params, **fixed_params)

        q_values = np.zeros((len(behav), 3))  # 3 targets
        stay_probas = np.zeros(len(behav))
        pes = np.zeros(len(behav))
        session_previous = 'none'
        for i, (_, row) in enumerate(behav.iterrows()):
            # If we are in a new session, reset the agent
            session_curr = row["session"]
            if session_curr != session_previous:
                agent.reset()
                session_previous = session_curr

            # Get the action and reward taken by the agent
            action, reward = row["target"], row["feedback"]
            
            # Get latent value function
            q_values[i] = agent.q_values
            stay_probas[i] = np.nan  # agent._get_stay_proba()  # TODO: implement stay probability for RL agent

            # Skip updating if NaN actions (interrupted trials)
            if np.isnan(action):
                #q_values[i] = np.nan
                #stay_probas[i] = np.nan
                continue

            # update the agent
            agent.update_values(int(action) - 1, int(reward))
            # save pe used for value updating
            pes[i] = agent.pe

        
        for j in range(3):
            behav.loc[:, f'Q_{j+1}'] = q_values[:, j]
        
        if add_proba:
            behav.loc[:, 'stay_proba'] = stay_probas
        if add_PE:
            behav.loc[:, 'PE'] = pes

        # write back the q_values to the original dataframe, matching the indices
        behavior.loc[behav.index, 'Q_1'] = behav.loc[behav.index, 'Q_1'].values
        behavior.loc[behav.index, 'Q_2'] = behav.loc[behav.index, 'Q_2'].values
        behavior.loc[behav.index, 'Q_3'] = behav.loc[behav.index, 'Q_3'].values
        if add_proba:
            behavior.loc[behav.index, 'stay_proba'] = behav.loc[behav.index, 'stay_proba'].values
        if add_PE:
            behavior.loc[behav.index, 'PE_RL'] = behav.loc[behav.index, 'PE'].values

    if digitize:
        # Create bin edges (from min to max value, or 0-1 if specified)
        all_qs = behavior[['Q_1', 'Q_2', 'Q_3']].values.flatten()
        all_qs = all_qs[~np.isnan(all_qs)]
        min_Q, max_Q = np.min(all_qs), np.max(all_qs)
        bin_edges = np.linspace(min_Q, max_Q+1e-5, n_classes+1)

        # Perform the binning
        for j in range(3):
            behavior[f'Q_{j+1}'] = np.digitize(
                behavior[f'Q_{j+1}'], 
                bins=bin_edges, 
                right=False) - 1
        
    return behavior

def add_switch_info(behav_full, add_trials_since_switch=False, flip_coding=False, relative_to='previous'):
    """
    Is the decision during this trial means a Switch from the previous trial.

    Add switch info to behav.

    If relative to='previous', a trial is assigned with a True value if the current target is the same as the previous.
    If relative to='next', a trial is assigned with a True value if the current target is the same as the next one.
    """
    # sort by monkey, session, block_id, trial_id
    #behav_full = behav_full.sort_values(['monkey', 'session', 'trial_id'])

    behavs = []
    for (monkey, session), behav in behav_full.groupby(['monkey', 'session']):
        # reset index of behav
        switches = [np.nan]  # first trial has no previous trial
        trials_since_switch = [0]
        targets = behav.target.values

        n_since_switch = 0
        for i in range(1, len(targets)):
            if np.isnan(targets[i]) or np.isnan(targets[i-1]):
                switches.append(np.nan)
                n_since_switch += 1
            elif targets[i] == targets[i-1]:
                switches.append(False)
                n_since_switch += 1
            else:
                switches.append(True)
                n_since_switch = 0

            trials_since_switch.append(n_since_switch)

        switches.append(np.nan)  # last trial has no next trial
        
        # add to the behav dataframe
        if relative_to == 'previous':
            behav['switch'] = switches[:-1]
        elif relative_to == 'next':
            behav['switch'] = switches[1:]
        if add_trials_since_switch:
            behav['trials_since_switch'] = trials_since_switch

        behavs.append(behav)

    # concatenate the behavs
    behav_full = pd.concat(behavs, ignore_index=True)
    behav_full['switch'] = behav_full['switch'].astype(float)
    if flip_coding:
        behav_full['switch'] = behav_full['switch'].apply(lambda x: not x if not pd.isna(x) else x)
    
    return behav_full

def add_trials_since_switch(behav):
    import warnings
    warnings.warn("This function is deprecated, use add_switch_info with add_trials_since_switch=True instead.")

    behav = add_switch_info(behav, add_trials_since_switch=True)
    return behav
