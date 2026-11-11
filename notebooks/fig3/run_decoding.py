"""
Time-resolved population decoding for all sessions.

Run: python run_decoding.py <config>, with <config> one of: fb_target, stay_value (Fig. 3f), multiple_alphas_ridge (Fig. 4g),
alternative_model_values (Supp. Fig. S6), stay_value_current_trial (Supp. Fig. S7c).
Results are saved in RESULTS_PATH/population_decoding/<config>.
"""
import os
import sys

# Limit native-threaded math libs per process (prevents CPU oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import datetime
import concurrent.futures
import traceback
import xarray as xr
import logging

from popy.io_tools import load_metadata
from popy.decoding.population_decoders import run_decoder
import popy.config as cfg


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

def save_results(xr, floc):
    xr.to_netcdf(os.path.join(floc, 'scores.nc'))

### Set parameters

CONFIGS = {
    # Fig. 3f: decoding feedback and target
    "fb_target": {
        "conditions": ['feedback', 'target'],
        "group_targets": None,
        "K_fold": 5,
        "step_len": 0.05,
        "n_perm": 500,
        "n_extra_trials": (0, 1),
        "floc": os.path.join(cfg.RESULTS_PATH, "population_decoding", "fb_target"),
        "msg": "Fig. 3f: decoding feedback and target.",
    },
    # Fig. 3f: decoding the stay value
    "stay_value": {
        "conditions": ['stay_value'],
        "group_targets": None,
        "K_fold": 10,
        "step_len": 0.05,
        "n_perm": 500,
        "n_extra_trials": (-1, 0),
        "floc": os.path.join(cfg.RESULTS_PATH, "population_decoding", "stay_value"),
        "msg": "Fig. 3f: decoding the stay value.",
    },
    # Fig. 4g: decoding the stay value computed with different learning rates
    "multiple_alphas_ridge": {
        "conditions": [f'stay_value_{alpha:.2f}' for alpha in np.arange(0.05, 1, 0.05)],
        "group_targets": None,
        "K_fold": 10,
        "step_len": 0.1,
        "n_perm": 500,
        "n_extra_trials": (0, 0),
        "floc": os.path.join(cfg.RESULTS_PATH, "population_decoding", "multiple_alphas_ridge"),
        "msg": "Fig. 4g: stay value with different learning rates.",
    },
    # Supp. Fig. S6: decoding the Q-values of the inferential and standard RL models (parameters of the run of the stored results)
    "alternative_model_values": {
        "conditions": ['stay_value', 'Q_1_inf', 'Q_1_stand', 'Q_2_inf', 'Q_2_stand', 'Q_3_inf', 'Q_3_stand', 'Q_chosen_inf', 'Q_chosen_stand'],
        "group_targets": None,
        "K_fold": 10,
        "step_len": 0.1,
        "n_perm": 100,
        "n_extra_trials": (-1, 0),
        "floc": os.path.join(cfg.RESULTS_PATH, "population_decoding", "alternative_model_values"),
        "msg": "Supp. Fig. S6: decoding the Q-values of the inferential and standard RL models.",
    },
    # Supp. Fig. S7c: decoding the stay value within the current trial (parameters of the run of the stored results)
    "stay_value_current_trial": {
        "conditions": ['stay_value'],
        "group_targets": None,
        "K_fold": None,  # None: default number of folds (10)
        "step_len": 0.05,
        "n_perm": 500,
        "n_extra_trials": (0, 0),
        "floc": os.path.join(cfg.RESULTS_PATH, "population_decoding", "stay_value_current_trial"),
        "msg": "Supp. Fig. S7c: decoding the stay value within the current trial.",
    },
}

### Run

if __name__ == '__main__':
    PARAMS = CONFIGS[sys.argv[1]]  # e.g. python run_decoding.py fb_target
    init_io(PARAMS)  # Initialize logging and create results folder

    monkeys, sessions = get_all_sessions()  # Get a pandas df containing all sessions' meta information
    
    n_cores = int(os.environ.get('POPY_N_CPUS', max(1, np.min([111, os.cpu_count()-3]))))  # number of parallel workers (set POPY_N_CPUS to limit memory use)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # submit jobs
        futures, future_proxy_mapping = [], {}
        for monkey, session in zip(monkeys, sessions):
            future = executor.submit(run_decoder, monkey, session, PARAMS, load_data=False, save_data=False)  # Run decoder for each session
            futures.append(future)
            future_proxy_mapping[future] = (monkey, session)

        # wait for results, save them
        count = 0
        xrs = []  
        for future in concurrent.futures.as_completed(futures):
            try:
                res, session_log = future.result()
                monkey_fut, session_fut = future_proxy_mapping[future]

                # Append results to existing results and save after each session
                if len(xrs) == 0:  # First result - save directly
                    xrs = res
                else:  # Not first result - concatenate to existing results
                    xrs = xr.concat([xrs, res], dim='session')
            
                # Save results after each session
                save_results(xrs, PARAMS['floc'])  # Save results after each session

                # Log progress
                for line in session_log:
                    logging.info(line)
                logging.info(f"Finished for monkey {monkey_fut} and session {session_fut}")

                # print log
                print(f'Progress: {count+1}/{len(monkeys)}')
                count += 1

            except Exception as e:  # Catch exceptions and log them
                logging.error(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}")
                print(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}\n")
                traceback.print_exc()  # Print traceback (?)

    end_log()
    print(f'Finished all on {datetime.datetime.now()}')
