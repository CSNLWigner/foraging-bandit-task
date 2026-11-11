# @title Imports
import os

# Limit native-threaded math libs per process (prevents CPU oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gymnasium as gym
import pandas as pd
from skopt.space import Real
from multiprocessing import Pool
from tqdm import tqdm

from popy.config import RESULTS_PATH
from popy.simulation_tools import *
from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.plotting.plotting_tools import *

from popy.simulation_helpers import *

def res_to_dataframe(results):
    res_all = pd.DataFrame.from_dict(results, orient='index').reset_index()  # convert to dataframe
    res_all = res_all.rename(columns={'index': 'Model'})  # rename index column to Model

    cols = list(res_all.columns)
    first_columns = ['Model', 'epsilon', 'alpha', 'alpha_unchosen', 'beta', 'V0', 
                     'forgetting_rate', 'forgetting_threshold', 
                     'stickiness_bias', 
                     'b1', 'b2', 'b3', 
                     'abandoned_bias', 'abandoned_decay']
    present = [c for c in first_columns if c in cols]
    rest = [c for c in cols if c not in present]
    new_columns = present + rest

    res_all = res_all[new_columns]

    return res_all

def behavs_to_dataframe(behaviors_simulated):
    # combine all behaviors into one dataframe
    behavs = []

    # then add the simulated behaviors
    for key, behav_temp in behaviors_simulated.items():
        behav_temp['session'] = 0
        behav_temp['model'] = key
        behavs.append(behav_temp)

    behaviors_simulated_all = pd.concat(behavs, axis=0)

    # reorder columns
    cols = ['model', 'session'] + [col for col in behav_temp.columns if col not in ['session', 'model']]
    behaviors_simulated_all = behaviors_simulated_all[cols]

    # reset index
    behaviors_simulated_all = behaviors_simulated_all.reset_index(drop=True)

    return behaviors_simulated_all

def save_res_and_behav(results, behaviors_simulated, floc):
    res_all = res_to_dataframe(results)
    floc_res_temp = os.path.join(floc, f'optimization_results.csv')
    res_all.to_csv(floc_res_temp, index=False)

    # save behaviors
    behaviors_simulated_all = behavs_to_dataframe(behaviors_simulated)
    floc_simulations_temp = os.path.join(floc, f'simulation_behaviors.pkl')
    behaviors_simulated_all.to_pickle(floc_simulations_temp)


# Analysis parameters
ANALYSIS_PARAMETERS = {
    "n_calls": 400,
    "n_initial_points": 250,
    "n_simulation_trials": 100_000,
    "n_cpus": max(1, os.cpu_count() - 2),  # leave some CPUs free
    "verbose": False,
    "make_plots": False,
    "CV_splits": None
}

floc = os.path.join(RESULTS_PATH, 'behav_modeling', 'optimizing')

# Fit parameters
fit_params = {
    'epsilon': Real(.0, .5, name='epsilon'),
    'alpha': Real(0.01, 1, name='alpha'),
    'beta': Real(10, 100, name='beta'),
    'stickiness_bias': Real(0.0, 50.0, name='stickiness_bias'),
    'V0': Real(0.0001, .4, name='V0'),
}

# Model configurations
MODELS = {
    "WSLS": {
        "agent_class": WSLSAgent,
        "fixed_params": {},
        "free_params": ["epsilon"],
    },
    "Standard RL - stickiness": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": False},
        "free_params": ["alpha", "beta", "stickiness_bias"],
    },
    "Inferential RL - stickiness": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": True},
        "free_params": ["alpha", "beta", "stickiness_bias"],
    },
    "Foraging": {
        "agent_class": ForagingAgent,
        "fixed_params": {"reset_on_switch": True},
        "free_params": ["alpha", "beta", "V0"],
    },
    "HSMM": {
        "agent_class": HSMMAgent,
        "fixed_params": {},
        "free_params": ["beta"],
    },
}


def _run_model_optimization(model_name, env):
    """Optimize a single model and return results."""
    model_params = MODELS[model_name]

    agent_class = model_params["agent_class"]
    fixed_params = model_params["fixed_params"]
    free_params = model_params["free_params"]
    param_space = [fit_params[p] for p in free_params]
    
    res_temp = fit_agent(
        agent_class, 
        param_space, 
        env, 
        fixed_params=fixed_params, 
        fit_on='rr', 
        make_plots=ANALYSIS_PARAMETERS['make_plots'], 
        verbose=ANALYSIS_PARAMETERS['verbose'], 
        n_calls=ANALYSIS_PARAMETERS['n_calls'],
        n_initial_points=ANALYSIS_PARAMETERS['n_initial_points'],
        n_jobs=1
    )
    
    res = res_temp['best_params']  # Add best parameters to results
    
    # Simulate behavior with best parameters
    behavior_simulated = simulate_agent(
        agent_class=agent_class, 
        params=res_temp['best_params'],
        env=env, 
        fixed_params=fixed_params, 
    )
    reward_rate = behavior_simulated['reward'].mean()
    proba_best = (behavior_simulated['action'] == behavior_simulated['best_arm']).mean()

    res['Reward rate'] = reward_rate
    res['Proba best'] = proba_best
    
    return {model_name: pd.Series(res)}, {model_name: behavior_simulated}


def _run_model_optimization_star(args):
    """Unpack args for Pool.imap/imap_unordered."""
    return _run_model_optimization(*args)


if __name__ == '__main__':
    # Setup
    env = gym.make(
        "zsombi/monkey-bandit-task-v0", 
        n_arms=3, 
        max_episode_steps=ANALYSIS_PARAMETERS['n_simulation_trials']
    )
    
    results = {}
    behaviors_simulated = {}

    # Prepare tasks for parallel processing
    tasks = [(model_name, env) for model_name in MODELS.keys()]

    # Run optimization in parallel
    with Pool(ANALYSIS_PARAMETERS['n_cpus']) as pool:
        optimization_results = list(
            tqdm(
                pool.imap_unordered(_run_model_optimization_star, tasks),
                total=len(tasks),
                desc="Optimizing models",
                unit="model",
            )
        )

    # Collect results
    for res_dict, behav_dict in optimization_results:
        results.update(res_dict)
        behaviors_simulated.update(behav_dict)

    # Save results
    os.makedirs(floc, exist_ok=True)
    save_res_and_behav(results, behaviors_simulated, floc)
    print('Optimization completed and results saved')
