""" Run code (in the background):
nohup python -u run_model_recovery.py > run_model_recovery.log 2>&1 &   # -> results/behav_modeling/recovery/model_recovery_10k.csv (~13 h on 20 CPUs)

Model recovery (Supplementary Fig. 3) and parameter recovery (Supplementary Fig. 4, the fits with the true model).
"""
import os

# Limit native-threaded math libs per process (prevents CPU oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import numpy as np
import pandas as pd
import gymnasium as gym
from skopt.space import Real
from multiprocessing import Pool, cpu_count
import datetime
from tqdm.auto import tqdm


# Import agents and utilities from the project
from popy.simulation_tools import *  # Provides agents like ForagingAgent, QLearner
from popy.simulation_helpers import simulate_agent, fit_simulate
from popy.config import RESULTS_PATH

def sample_beta(scale):
    while True:
        beta = np.random.exponential(scale=scale)

        if beta > 1 and beta < 20: # only keep betas in this range
            return beta

ANALYSIS_PARAMETERS = {
    "T": 10_000,  # Total trials per simulation (same order of magnitude as real data)
    "N": 100,  # Number of simulations to run
    "n_calls": 150,  # 200,  # Number of optimizer calls
    "n_initial_points": 50,  # 100,  # Number of initial points for optimizer
    "n_cpus": max(1, cpu_count() - 4),  # Number of CPUs to use
    "output_file": "model_recovery_10k.csv",
}

gen_params = {
    "epsilon": lambda: np.random.uniform(0, .5),
    "alpha": lambda: np.random.uniform(0.1, 0.7),
    "beta": lambda: sample_beta(scale=5.0),
    "V0": lambda: np.random.uniform(0.05, 0.5),
    "stickiness_bias": lambda: np.random.exponential(1),
}

fit_params = {
    "epsilon": Real(0.01, 1, name='epsilon'),
    "alpha": Real(0.01, 0.9, name="alpha"),
    "beta": Real(0.05, 50.0, name="beta"),
    "stickiness_bias": Real(0.0, 10.0, name="stickiness_bias"),
    "V0": Real(0.05, 0.7, name="V0"),
}

MODELS = {
    "Foraging": {
        "agent_class": ForagingAgent,
        "fixed_params": {"reset_on_switch": True},
        "free_params": ["alpha", "beta", "V0"],
    },
    "Inferential RL - stickiness": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": True},
        "free_params": ["alpha", "beta", "stickiness_bias"],
    },
    "Standard RL - stickiness": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": False},
        "free_params": ["alpha", "beta", "stickiness_bias"],
    },
}

def make_env():
    """Create the monkey-bandit environment used for simulations."""
    return gym.make("zsombi/monkey-bandit-task-v0", n_arms=3, max_episode_steps=ANALYSIS_PARAMETERS['T'])


def _run_model_recovery(sim_model, fit_model, iteration):
    """Helper function to run a single model recovery iteration.
    
    sim_model: true model used for simulation
    fit_model: true model used for fitting
    iteration: iteration number (used to generate same simulation for all fit models)
    """
    start_time = time.time()
    #print(f"[{sim_model} -----> {fit_model}], iteration {iteration+1}/{ANALYSIS_PARAMETERS['N']}", flush=True)
    
    # Create environment for this process
    env = make_env()

    # Create seed to ensure same simulation for all fit models
    sim_model_index = list(MODELS.keys()).index(sim_model)
    sim_seed = sim_model_index * 1000 + iteration
    
    ## 1: SIMULATE BEHAVIOR
    sim_agent_class = MODELS[sim_model]['agent_class']
    sim_fixed_params = MODELS[sim_model]['fixed_params']
    np.random.seed(sim_seed)
    sim_free_params = MODELS[sim_model]['free_params']
    sim_params_true = {free_param: gen_params[free_param]() for free_param in sim_free_params}

    behavior_simulated = simulate_agent(
        agent_class=sim_agent_class,
        params=sim_params_true,
        env=env,
        fixed_params=sim_fixed_params,
    )
    behavior_simulated['monkey'] = 'simulated_agent'
    behavior_simulated['session'] = 0

    ## 2: FIT THE SIMULATED BEHAVIOR
    fit_agent_class = MODELS[fit_model]['agent_class']
    fit_fixed_params = MODELS[fit_model]['fixed_params']
    fit_free_params = MODELS[fit_model]['free_params']
    fit_param_space = [fit_params[free_param] for free_param in fit_free_params]  # search space for optimizer


    
    # Fit the specified model to the simulated data
    try:    
        results, _ = fit_simulate(
            fit_agent_class,
            fit_param_space,
            env,
            behavior_simulated,
            fixed_params=fit_fixed_params,
            CV_splits=False,
            make_plots=False,
            n_calls=ANALYSIS_PARAMETERS['n_calls'],
            n_initial_points=ANALYSIS_PARAMETERS['n_initial_points'],
            n_jobs=1,
            verbose=False,
        )

        result_dict = {
            'model_true': sim_model,
            'model_fitted': fit_model,
            'iteration': iteration,
            **(sim_params_true if sim_params_true is not None else {}),
            **{f'{param}_recovered': results[param] for param in fit_free_params},
            'LPT_best': results['LPT_best'],
        }
        return result_dict

    except Exception as e:
        print(f"Error fitting model {fit_model} to data simulated from {sim_model} with params {sim_params_true}: {e}")
        # Return NaNs for all recovered parameters and LPT_best
        result_dict = {
            'model_true': sim_model,
            'model_fitted': fit_model,
            'iteration': iteration,
            **(sim_params_true if sim_params_true is not None else {}),
            **{f'{param}_recovered': np.nan for param in fit_free_params},
            'LPT_best': np.nan,
        }
        return result_dict


def _run_model_recovery_star(args):
    """Unpack args for Pool.imap/imap_unordered."""
    return _run_model_recovery(*args)


def generate_model_recovery(env, output_csv_path: str):
    """
    Run model recovery for the models (all combinations).
    Saves a CSV with LPT for each run for each true model (used for simulation) and each fitted model (used for fitting). Also saves the true parameters of the true model.
    """
    n_models = len(MODELS)
    n_tasks = ANALYSIS_PARAMETERS['N'] * (n_models ** 2)
    print(f"[Model recovery] Running {n_tasks} fits ({ANALYSIS_PARAMETERS['N']} sims × {n_models}×{n_models} models) on {ANALYSIS_PARAMETERS['n_cpus']}/{os.cpu_count()} CPUs")
    
    # Create task list: for each iteration, fit all models to the same simulation
    tasks = []
    for iteration in range(ANALYSIS_PARAMETERS['N']):
        for sim_model in MODELS.keys():
            for fit_model in MODELS.keys():
                tasks.append((sim_model, fit_model, iteration))
    
    with Pool(ANALYSIS_PARAMETERS['n_cpus']) as pool:
        recovery_results = list(
            tqdm(
                pool.imap_unordered(_run_model_recovery_star, tasks),
                total=len(tasks),
                desc="Model recovery",
                unit="fit",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]",
            )
        )

    recovery_df = pd.DataFrame(recovery_results)
    recovery_df.to_csv(output_csv_path, index=False)
    print(f"[Model recovery] Completed {len(recovery_results)} simulations. Saved to {output_csv_path}")


if __name__ == "__main__":
    env = make_env()
    os.makedirs(os.path.join(RESULTS_PATH, "behav_modeling", "recovery"), exist_ok=True)
    generate_model_recovery(
        env,
        output_csv_path=os.path.join(RESULTS_PATH, "behav_modeling", "recovery", ANALYSIS_PARAMETERS['output_file']),
    )
