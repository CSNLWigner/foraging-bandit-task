
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
import time
from multiprocessing import Pool
from functools import partial

from popy.simulation_tools import *
from popy.io_tools import load_behavior
from popy.behavior_data_tools import *
from popy.simulation_helpers import fit_agent
from popy.config import RESULTS_PATH

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

fit_params = {
    "epsilon": Real(0.01, 1, name='epsilon'),
    "alpha": Real(0.01, 0.9, name="alpha"),
    "beta": Real(0.05, 50.0, name="beta"),
    "stickiness_bias": Real(0.0, 10.0, name="stickiness_bias"),
    "V0": Real(0.05, 0.7, name="V0"),
}

gp_params = {
    "n_calls": 350,
    "n_initial_points": 100,
    "n_jobs": 1,
    "verbose": False,
}

n_bootstrap = 100
n_cpus = max(1, os.cpu_count() - 4)  # leave some CPUs free

def bootstrap_sessions(behav, bootstrap_idx=None):
    """
    Bootstrap sessions from the original data with unique IDs for each copy.
    
    Each sampled session gets a unique ID: 
    - "session_id_copy0", "session_id_copy1", etc. (or with bootstrap_idx if provided)
    """

    # set random seed to idx
    if bootstrap_idx is not None:
        np.random.seed(bootstrap_idx)

    session_ids = behav["session"].unique()
    bootstrapped_sessions = np.random.choice(session_ids, size=len(session_ids), replace=True)
    
    # Track how many times each session has been selected
    session_count = {}
    data_chunks = []
    
    for orig_sid in bootstrapped_sessions:
        if orig_sid not in session_count:
            session_count[orig_sid] = 0
        else:
            session_count[orig_sid] += 1
        
        # Get the data for this session
        session_data = behav[behav["session"] == orig_sid].copy()
        
        # Create unique ID for this copy
        if bootstrap_idx is not None:
            new_id = f"{orig_sid}_boot{bootstrap_idx}_copy{session_count[orig_sid]}"
        else:
            new_id = f"{orig_sid}_copy{session_count[orig_sid]}"
        
        # Update session ID to the unique identifier
        session_data["session"] = new_id
        data_chunks.append(session_data)
    
    bootstrapped_behav = pd.concat(data_chunks, ignore_index=True)
    return bootstrapped_behav


def _run_single_bootstrap(
    bootstrap_idx,
    agent_class,
    param_space,
    env,
    behav_data,
    fixed_params,
    model_name,
    gp_params,
):
    """
    Run a single bootstrap fitting iteration.
    
    Parameters:
    -----------
    bootstrap_idx : int
        Bootstrap iteration index
    agent_class : class
        Agent class to fit
    param_space : list
        Parameter space for optimization
    env : gym.Env
        Environment
    behav_data : pd.DataFrame
        Original behavioral data
    fixed_params : dict
        Fixed parameters for agent
    model_name : str
        Name of the model
    gp_params : dict
        Parameters for GP minimize
        
    Returns:
    --------
    row : dict
        Dictionary with results for this bootstrap iteration
    """
    # Bootstrap the data
    bootstrapped_behav = bootstrap_sessions(behav_data, bootstrap_idx=bootstrap_idx)

    # Fit the model
    start_time = time.time()
    result = fit_agent(
        agent_class=agent_class,
        param_space=param_space,
        env=env,
        behav_data=bootstrapped_behav,
        fixed_params=fixed_params,
        fit_on="ll",
        n_calls=gp_params["n_calls"],
        n_initial_points=gp_params["n_initial_points"],
        n_jobs=gp_params["n_jobs"],
        verbose=gp_params["verbose"],
    )
    elapsed_time = time.time() - start_time
    
    # Flatten the results dictionary
    row = {
        "model": model_name,
        "bootstrap_idx": bootstrap_idx,
    }
    
    # Add parameters (unpacked from best_params dict)
    if isinstance(result["best_params"], dict):
        for param_name, param_value in result["best_params"].items():
            row[param_name] = param_value
    
    # Add results metrics
    row["best_ll"] = result["best_ll"]
    row["bic"] = result["bic"]
    row["lpt"] = result.get("lpt", None)
    row["fit_time_sec"] = elapsed_time
    
    print(f"Bootstrap {bootstrap_idx + 1}: LL={result['best_ll']:.4f}, BIC={result['bic']:.4f}, Time={elapsed_time:.2f}s")
    
    return row

# # Simulate models

# ## gp_minimize

def run_bootstrap_fitting(
    agent_class,
    param_space,
    env,
    behav_data,
    fixed_params,
    model_name,
    n_bootstrap=10,
    gp_params=None,
    output_csv=None,
    n_cpus=1,
):
    """
    Run model fitting N times on bootstrapped data and save results to CSV.
    
    Parameters:
    -----------
    agent_class : class
        Agent class to fit
    param_space : list
        Parameter space for optimization
    env : gym.Env
        Environment
    behav_data : pd.DataFrame
        Original behavioral data
    fixed_params : dict
        Fixed parameters for agent
    model_name : str
        Name of the model (for results tracking)
    n_bootstrap : int
        Number of bootstrap iterations
    gp_params : dict, optional
        Parameters for GP minimize (n_calls, n_initial_points, n_jobs, verbose)
    output_csv : str, optional
        Path to save results CSV. If None, does not save individual CSV (useful for batch processing).
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with all fitting results
    """
    
    if gp_params is None:
        gp_params = {
            "n_calls": 100,
            "n_initial_points": 50,
            "n_jobs": -1,
            "verbose": False,
        }
    
    results_list = []
    
    print(f"Starting bootstrap fitting for model: {model_name}")
    print(f"Running {n_bootstrap} bootstrap iterations on {n_cpus} CPUs...\n")
    
    # Create partial function with fixed parameters
    fit_single = partial(
        _run_single_bootstrap,
        agent_class=agent_class,
        param_space=param_space,
        env=env,
        behav_data=behav_data,
        fixed_params=fixed_params,
        model_name=model_name,
        gp_params=gp_params,
    )
    
    # Run bootstrap iterations in parallel
    with Pool(processes=n_cpus) as pool:
        results_list = pool.map(fit_single, range(n_bootstrap))
    
    print()
    
    # Create DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Optionally save to CSV
    if output_csv is not None:
        results_df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to: {output_csv}")
        print(f"\nSummary statistics:")
        print(results_df[["best_ll", "bic", "lpt", "fit_time_sec"]].describe())
    
    return results_df

# Run bootstrap fitting for both monkeys
for monkey in ["ka", "po"]:
    print(f"\n\n{'#'*70}")
    print(f"# Processing Monkey: {monkey.upper()}")
    print(f"{'#'*70}\n")
    
    # Load data for this monkey
    env = gym.make("zsombi/monkey-bandit-task-v0", n_arms=3, max_episode_steps=100_000)
    behav_monkey = load_behavior(monkey)
    behav_monkey = drop_time_fields(behav_monkey)
    behav_monkey = add_switch_info(behav_monkey)
    behav_monkey = convert_column_format(behav_monkey, original='behavior')
    behav_monkey = behav_monkey.dropna()
    
    print(f"Loaded {len(behav_monkey)} trials from monkey {monkey}\n")
    
    # Run bootstrap fitting for all models
    all_results = []
    
    output_csv = os.path.join(
        RESULTS_PATH, 
        "behav_modeling", 
        "bootstrap", 
        f"{monkey}_bootstrap_all_models.csv"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    for model_name, model_config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Running bootstrap for: {model_name}")
        print(f"{'='*60}")
        
        agent_class = model_config["agent_class"]
        fixed_params = model_config["fixed_params"]
        free_params = model_config["free_params"]
        param_space = [fit_params[param] for param in free_params]
        
        # Run bootstrap fitting (returns dataframe, we'll append to list)
        results_df = run_bootstrap_fitting(
            agent_class=agent_class,
            param_space=param_space,
            env=env,
            behav_data=behav_monkey,
            fixed_params=fixed_params,
            model_name=model_name,
            n_bootstrap=n_bootstrap,
            gp_params=gp_params,
            output_csv=None,  # Don't save individual CSVs
            n_cpus=n_cpus,
        )
        
        all_results.append(results_df)
    
    # Combine all results and save to single CSV
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Monkey {monkey.upper()}: All results saved to: {output_csv}")
    print(f"{'='*60}")
    print(f"\nCombined summary statistics:")
    print(combined_results.groupby("model")[["best_ll", "bic", "lpt", "fit_time_sec"]].describe())

print(f"\n\n{'#'*70}")
print(f"# ✓ All monkeys processed successfully!")
print(f"{'#'*70}")


