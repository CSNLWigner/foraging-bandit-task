"""
Within- and across-target neural value: in each session, trials are grouped by the chosen target. For each target, a
logistic regression decoder of the previous outcome (R_1) is fit on the population activity, and the activity is projected
on its decision vector: the trials of that target with leave-one-out cross-validation (within target), the trials of the
other two targets with the decoder fit on all trials of that target (across target). The projections are averaged
within a time window: 1.5-3.5 s of the normalized trial for Fig. 5 and Supplementary Fig. S13 ('before_fb'), and the six
windows of Supplementary Fig. S14 (TIMES_OF_INTEREST; w1-w6 = after_fb, it1, it2, before_fb, tt_centered, tt_centered_short).

Run: python run_neural_value_across_target.py [window ...]  (default: all windows of TIMES_OF_INTEREST, one after the other;
     results saved in RESULTS_PATH/population_decoding/across_target_<window>)
"""
##% title Imports
import os

# Limit native-threaded math libs per process (prevents CPU oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import traceback
import concurrent.futures
import datetime
import logging

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.linear_model import LinearRegression, LogisticRegression

import popy.config as cfg
from popy.io_tools import load_metadata
from helpers_across_target import load_data_custom


PATH = cfg.PROJECT_PATH_LOCAL

##% data loading


def NEW_project_across_targets(neural_dataset, fit_subspace_on='R_1'):
    '''
    This is the modified version of the code, where we regress the neural data to the last outcome instead of the model based value.
    '''
    # Get a vector of selected targets [0, 1, 2] for each trial
    target_vector = neural_dataset['target'].values

    # write back to xarray, preserve the trial and time dimensions and corresponding coordinates
    time_coords = {name: coord for name, coord in neural_dataset.coords.items() if 'time' in coord.dims}
    trial_coords = {name: coord for name, coord in neural_dataset.coords.items() if 'trial_id' in coord.dims}

    # create train and test splits
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)

    projections = np.full((3, len(neural_dataset.trial_id), len(neural_dataset.time)), np.nan)
    trial_ids = neural_dataset.trial_id.values
    time_bins = neural_dataset.time.values

    ## Part 1: project within target: leave one out method (train on all but one trial, project the left out trial to the subspace defined by the training trials)
    for t_id, t in enumerate(neural_dataset.time):
        # if t % 1 == 0:print(t_id)
        # first project within target with leave one out method
        for fold, target_temp in enumerate(np.unique(target_vector)):
            neural_dataset_within = neural_dataset.firing_rates.sel(trial_id=neural_dataset.target == target_temp, time=t)

            for test_trial in np.unique(neural_dataset_within.trial_id.values):
                train_trials = neural_dataset_within.trial_id.values != test_trial
                X_train = neural_dataset_within.sel(trial_id=train_trials).values
                y_train = neural_dataset_within.sel(trial_id=train_trials)[fit_subspace_on].values

                X_project = neural_dataset_within.sel(trial_id=test_trial).values

                # fit the model
                clf.fit(X_train, y_train)

                # project the dataset (at this point) to this subspace
                weights = clf.coef_.reshape(-1, 1)
                # trial_projected = np.dot(X_project, weights)
                weights_norm = np.linalg.norm(weights)
                trial_projected = (np.dot(X_project, weights) + clf.intercept_) / weights_norm

                trial_idx = np.where(trial_ids == test_trial)[0][0]
                projections[fold, trial_idx, t_id] = trial_projected.squeeze()

    ## Part 2: project alternative targets: train the model on one target, project the other targets to this subspace
    # scores = np.empty((2, len(neural_dataset.time)))
    for t_id, t in enumerate(neural_dataset.time):
        for fold, target_temp in enumerate(np.unique(target_vector)):
            neural_dataset_within = neural_dataset.firing_rates.sel(trial_id=neural_dataset.target == target_temp, time=t)
            neural_dataset_across = neural_dataset.firing_rates.sel(trial_id=neural_dataset.target != target_temp, time=t)

            # get the firing rates for the current time point
            X_train = neural_dataset_within.values
            y_train = neural_dataset_within[fit_subspace_on].values

            # fit the model
            clf.fit(X_train, y_train)

            # project the dataset (at this point) to this subspace
            weights = clf.coef_.reshape(-1, 1)  # reshape to ensure correct dimensions
            # trials_projected = np.dot(neural_dataset_across, weights)
            weights_norm = np.linalg.norm(weights)
            trials_projected = (np.dot(neural_dataset_across, weights) + clf.intercept_) / weights_norm

            trial_idxs = [np.where(trial_ids == trial_id)[0][0] for trial_id in neural_dataset_across.trial_id.values]
            projections[fold, trial_idxs, t_id] = trials_projected.squeeze()

    # Create a DataArray with the projected data
    data_projected_da = xr.DataArray(projections, dims=('subspace', 'trial_id', 'time'),
                                      coords={'subspace': [f'target_{int(temp)}' for temp in np.unique(target_vector)],
                                              'trial_id': neural_dataset.trial_id.values,
                                              'time': neural_dataset.time.values})
    # Add the original coordinates
    for name, coord in time_coords.items():
        data_projected_da = data_projected_da.assign_coords({name: coord})
    for name, coord in trial_coords.items():
        data_projected_da = data_projected_da.assign_coords({name: coord})

    return data_projected_da


def extract_neural_data(neural_values, data_projected):
    results = []

    for trial_id in np.unique(data_projected.trial_id.values):
        neural_values_curr = neural_values.sel(trial_id=trial_id)
        if trial_id + 1 not in neural_values.trial_id.values:
            continue
        neural_values_p1 = neural_values.sel(trial_id=trial_id + 1)

        target_curr = float(neural_values_curr.target.values)
        target_p1 = float(neural_values_p1.target.values)
        if target_curr != target_p1:
            continue
        fb = int(neural_values_curr.feedback.values)

        #value_same = neural_values_curr.sel(subspace=f'target_{str(int(target_curr))}').data

        df_temp = {'trial_id': trial_id, 
                    'target': target_curr, 'feedback': fb}

        for subspace in neural_values_curr.subspace.values:
            df_temp_subspace = df_temp.copy()

            if subspace == f'target_{str(int(target_curr))}':  # TODO: az xarrayben nem így van elnevzve
                df_temp_subspace['type'] = 'within'
            else:
                df_temp_subspace['type'] = 'alter'

            V_t = neural_values_curr.sel(subspace=subspace).data
            V_t_p1 = neural_values_p1.sel(subspace=subspace).data
            dV = - V_t + V_t_p1

            df_temp_subspace[f'V_neural'] = float(V_t)
            df_temp_subspace[f'V_neural_p1'] = float(V_t_p1)
            df_temp_subspace[f'dV_neural'] = dV

            '''df_temp_subspace[f'V_behav'] = neural_values_curr.stay_value.values
            df_temp_subspace[f'V_behav_p1'] = neural_values_p1.stay_value.values
            df_temp_subspace[f'dV_behav'] = df_temp_subspace[f'V_behav'] - df_temp_subspace[f'V_behav_p1']'''

            results.append(df_temp_subspace)

    # convert to df
    results_df = pd.DataFrame(results)
    return results_df


### Load metadata

def get_all_sessions():
    session_metadata = load_metadata()
    session_metadata = session_metadata[session_metadata['block_len_valid'] == True]  # Only use sessions with valid block length

    monkeys = session_metadata['monkey'].values 
    sessions = session_metadata['session'].values

    return monkeys, sessions


### Configure logging

def end_log():
    # start time is the first log entry
    end_time = datetime.datetime.now()
    logging.info(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

def init_io(PARAMS):
    os.makedirs(PARAMS['floc'], exist_ok=True)

    # configure logging
    logging.basicConfig(filename=os.path.join(PARAMS['floc'], 'log.txt'),
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w')  # 'w' mode will overwrite the log file

    start_time = datetime.datetime.now()
    logging.info("PARAMS:")
    for key, value in PARAMS.items():
        logging.info(f'{key}: {value}')
    logging.info(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

### Save results

def save_results(dfs_all, floc):
    dfs_all.to_pickle(os.path.join(floc, 'behav.pkl'))

### Set parameters
 
TIMES_OF_INTEREST = {
    'after_fb':   (-4, -2),
    'it1':        (-2,  0),
    'it2':        (-.5,  1.5),
    'before_fb':  (1.5, 3.5),
    'tt_centered': (2,  3),    
    'tt_centered_short': (2.25,  2.75),    
}

FLOC_FOLDER = 'across_target_'  # TODO
N_FIRST_ONLY = None  # TODO: whether to use only the first trial of each block (to avoid contamination from previous trial). None or an int.

def make_params(key, time_window):
    FLOC_BASE = os.path.join(cfg.RESULTS_PATH, 'population_decoding')
    return {
        'time_of_interest': list(time_window),
        'floc': os.path.join(FLOC_BASE, FLOC_FOLDER+key),
        'n_first_only': N_FIRST_ONLY,  # whether to use only the first trial of each block (to avoid contamination from previous trial) 
    }

### Run

def run(monkey, session, PARAMS):
    print('running: ', monkey, session)
    #monkey, session, subregion = 'ka', '210322', 'vLPFC'

    time_of_interest = PARAMS['time_of_interest']

    neural_dataset = load_data_custom(monkey, session, area=None, n_extra_trials=(-1, 0), n_first_only=PARAMS['n_first_only'])  # load data for the current session, with all areas and one extra trial before the first trial (for projection purposes)

    dfs = []
    for subregion in np.unique(neural_dataset.subregion.data):
        # get data for current subregion and time window of interest
        neural_dataset_temp = neural_dataset.sel(unit=neural_dataset.subregion==subregion)
        neural_dataset_temp = neural_dataset_temp.sel(time=slice(*time_of_interest))

        # project the neural data across targets - from high dimensional space to 3 dimensional space
        data_projected = NEW_project_across_targets(neural_dataset_temp, fit_subspace_on='R_1')  
        
        # Extract 'neural values' for the specified time range by averaging across time points
        neural_values = data_projected.mean(dim='time')

        # Extract neural values and their change

        results_df = extract_neural_data(neural_values, data_projected)

        # add subregion info
        results_df['monkey'] = monkey
        results_df['session'] = session
        results_df['subregion'] = subregion

        dfs.append(results_df)
    dfs = pd.concat(dfs, ignore_index=True)
        
    session_log = []

    dfs = dfs.sort_values(by=['monkey', 'session', 'subregion', 'trial_id'])
    return dfs, session_log


if __name__ == '__main__':
    # run the same thing for different time windows of interest
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(TIMES_OF_INTEREST)  # optional: names of the windows to run
    for key, time_window in [(key, TIMES_OF_INTEREST[key]) for key in keys]:
        PARAMS = make_params(key, time_window)  # generate parameters for the current time window
        init_io(PARAMS)  # Initialize logging and create results folder

        monkeys, sessions = get_all_sessions()  # Get a pandas df containing all sessions' meta information
        
        n_cores = int(os.environ.get('POPY_N_CPUS', np.max([1, os.cpu_count()-2])))  # number of parallel workers (set POPY_N_CPUS to limit memory use)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            # submit jobs
            futures, future_proxy_mapping = [], {}
            for monkey, session in zip(monkeys, sessions):
                future = executor.submit(run, monkey, session, PARAMS)  # Run decoder for each session
                futures.append(future)
                future_proxy_mapping[future] = (monkey, session)

            # wait for results, save them
            count_good = 0
            count_bad = 0

            dfs_all = []  
            for future in concurrent.futures.as_completed(futures):
                try:
                    res, session_log = future.result()
                    monkey_fut, session_fut = future_proxy_mapping[future]

                    # Append results to existing results and save after each session
                    if len(dfs_all) == 0:
                        dfs_all = res
                    else:
                        dfs_all = pd.concat([dfs_all, res], ignore_index=True)
                
                    # Save results after each session
                    save_results(dfs_all, PARAMS['floc'])  # Save results after each session

                    # Log progress
                    for line in session_log:
                        logging.info(line)
                    logging.info(f"Finished for monkey {monkey_fut} and session {session_fut}")

                    count_good += 1

                except Exception as e:  # Catch exceptions and log them
                    logging.error(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}")
                    print(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}\n")
                    traceback.print_exc()  # Print traceback (?)

                    count_bad += 1

                print(f'Progress: {count_good + count_bad}/{len(monkeys)} failed: {count_bad}')


        end_log()
        print(f'Finished all on {datetime.datetime.now()}')
