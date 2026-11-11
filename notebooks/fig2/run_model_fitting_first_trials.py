# # Summary
#
# Fit the models to the first trials of each block only (Supplementary Fig. 5): same as run_model_fitting.py, but the likelihood
# is computed on the first N_FIRST trials of each block (fit on the whole data set, no cross-validation).

""" Run code (in the background):
nohup python -u run_model_fitting_first_trials.py 10 > run_model_fitting_first10.log 2>&1 &   # -> results/behav_modeling/fitting_first10
nohup python -u run_model_fitting_first_trials.py 15 > run_model_fitting_first15.log 2>&1 &   # -> results/behav_modeling/fitting_first15
"""

# @title Imports
import os
import sys

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


from popy.simulation_tools import *
from popy.io_tools import load_behavior, load_behavior_yuri
from popy.behavior_data_tools import *
from popy.plotting.plotting_tools import *
from popy.config import RESULTS_PATH

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
from popy.simulation_helpers import estimate_ll, estimate_rr, simulate_agent, cross_val_fit

def get_data_custom(monkey):
    if monkey in ['ka', 'po']:
        behav_monkey = load_behavior(monkey)
        behav_monkey = drop_time_fields(behav_monkey)
    elif monkey in ['yu_sham', 'yu_DCZ']:
        behav_monkey = load_behavior_yuri()
        behav_monkey = behav_monkey.loc[behav_monkey['monkey'] == monkey]
    else:
        raise ValueError(f'Unknown monkey: {monkey}')
    behav_monkey = add_switch_info(behav_monkey)
    behav_monkey = convert_column_format(behav_monkey, original='behavior')

    behav_monkey = behav_monkey.dropna()

    return behav_monkey

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

    return res_all[new_columns]

def behavs_to_dataframe(behaviors_simulated, monkey):
    # combine all behaviors into one dataframe
    behavs = []

    # Process simulations: start with the real behavior
    behav_monkey = behaviors_simulated[f'MONKEY {monkey.upper()}']
    behav_monkey = behav_monkey.drop(columns=['switch'])
    behav_monkey['model'] = 'recording'
    behavs.append(behav_monkey)

    # then add the simulated behaviors
    for key, behav_temp in behaviors_simulated.items():
        if key != f'MONKEY {monkey.upper()}':
            behav_temp['monkey'] = monkey
            behav_temp['session'] = 0
            behav_temp['model'] = key
            behavs.append(behav_temp)

    behaviors_simulated_all = pd.concat(behavs, axis=0)

    # reorder columns
    cols = ['monkey', 'model', 'session'] + [col for col in behav_monkey.columns if col not in ['monkey', 'session', 'model']]
    behaviors_simulated_all = behaviors_simulated_all[cols]

    # reset index
    behaviors_simulated_all = behaviors_simulated_all.reset_index(drop=True)

    return behaviors_simulated_all

def save_res_and_behav(results, behaviors_simulated, monkey, floc):
    res_all = res_to_dataframe(results)
    floc_res_temp = os.path.join(floc, f'simulation_results_{monkey}.csv')
    res_all.to_csv(floc_res_temp, index=False)

    # save behaviors
    behaviors_simulated_all = behavs_to_dataframe(behaviors_simulated, monkey)
    floc_simulations_temp = os.path.join(floc, f'simulation_behaviors_{monkey}.pkl')
    behaviors_simulated_all.to_pickle(floc_simulations_temp)

# fit_agent() and fit_simulate() of popy/simulation_helpers.py, with one change: the number of trials at the start of each block
# used by the 'll_first10' objective is the parameter n_first (10 in popy). The original analysis used the same code with 10
# trials (git 56f6df4) and with 15 trials (git 517a708, where it was called 'll_first15').

### Functions for fitting agents and cross-validation ###


def fit_agent(
    agent_class,
    param_space,
    env,
    behav_data=None,
    fixed_params=None,
    fit_on="ll",
    n_calls=50,
    n_initial_points=10,
    n_jobs=-1,
    verbose=False,
    make_plots=False,
    n_first=10,
):
    """
    Fit an agent model to behavioral data using Bayesian optimization.
    """
    if fixed_params is None:
        fixed_params = {}

    # Define the objective function with dynamic parameter handling
    @use_named_args(param_space)
    def objective(**params):
        if fit_on == 'll':
            # Run simulation and get log likelihood
            ll = estimate_ll(agent_class, params, behav_data, fixed_params, return_with_behav=False)
            return -ll
        elif fit_on == 'll_first10':
            # Run simulation and get log likelihood
            ll_all = estimate_ll(agent_class, params, behav_data, fixed_params, return_with_behav=True)
            ll_first10 = ll_all.groupby(['session', 'block_id']).head(n_first)['logp_action'].sum()
            return -ll_first10
        elif fit_on == 'rr':
            # Run simulation and get reward rate
            rr = estimate_rr(agent_class, params, env, fixed_params)
            return -rr
        else:
            raise ValueError(f"Invalid value for 'fit_on': {fit_on}")

    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        param_space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        verbose=verbose,
        acq_optimizer="lbfgs",
        n_jobs=n_jobs,
    )

    # Extract best parameters
    param_names = [p.name for p in param_space]
    best_params = dict(zip(param_names, result.x))

    # Create plots
    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Plot convergence
        plot_convergence(result, ax=axes[0])
        axes[0].set_title("Convergence Plot")

        # Plot parameter importance if there's more than one parameter
        if len(param_space) > 1:
            plot_objective(result, dimensions=param_names, ax=axes[1])
        else:
            plot_objective(result, ax=axes[1])
        axes[1].set_title("Parameter Importance")

        plt.tight_layout()

    # Compute additional metrics (LPT)
    if fit_on == 'll' or fit_on == 'll_first10':
        # Calculate BIC and log likelihood per trial
        best_ll = -result.fun  # Convert back to positive log likelihood
        n_params = len(param_space)
        n_trials = len(behav_data) if fit_on == 'll' else len(behav_data.groupby(['session', 'block_id']).head(n_first))
        bic = -2 * best_ll + n_params * np.log(n_trials)
        lpt = np.exp(best_ll / n_trials)

        return {
            #'result': result,
            'best_params': best_params,
            'best_ll': best_ll,
            'bic': bic,
            'lpt': lpt
        }
    elif fit_on == 'rr':
        return {
            #'result': result,
            'best_params': best_params,
            'best_reward_rate': -result.fun
        }
    else:
        raise ValueError(f"Invalid value for 'fit_on': {fit_on}")


### Main function for all analysis of model fitting in once ###


def fit_simulate(
    agent_class,
    param_space,
    env,
    behav,
    fixed_params=None,
    CV_splits=False,
    fit_on_first_10=False,
    n_calls=50,
    n_initial_points=10,
    n_jobs=-1,
    verbose=False,
    make_plots=False,
    n_first=10,
):
    """
    Fit an agent model to behavioral data and simulate the agent to get reward rate and probability of choosing best arm.
    """

    # container for the results
    res = {}

    #### Fit the model (CV) for performance ####
    if CV_splits:
        CV_scores = cross_val_fit(
            agent_class=agent_class,
            param_space=param_space,
            fixed_params=fixed_params,
            env=env,
            behav_data=behav,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            n_jobs=n_jobs,
        )

        res['LL_CV'] = CV_scores['LL']['mean']
        res['LL_std'] = CV_scores['LL']['std']
        res['BIC_CV'] = CV_scores['BIC']['mean']
        res['BIC_std'] = CV_scores['BIC']['std']
        res['LPT_CV'] = CV_scores['LPT']['mean']
        res['LPT_std'] = CV_scores['LPT']['std']

    #### Fit the model on full data for parameters ####
    if not fit_on_first_10:
        results = fit_agent(
            agent_class=agent_class,
            param_space=param_space,
            fixed_params=fixed_params,
            fit_on="ll",
            env=env,
            behav_data=behav,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            n_jobs=n_jobs,
            verbose=verbose,
            make_plots=make_plots,
        )
    else:
        results = fit_agent(
            agent_class=agent_class,
            param_space=param_space,
            fixed_params=fixed_params,
            fit_on='ll_first10',
            n_first=n_first,
            env=env,
            behav_data=behav,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            n_jobs=n_jobs,
            verbose=verbose,
            make_plots=make_plots,
        )
    
    # collect results
    res = {**res, **results['best_params']}  # Add best parameters to results

    res['LL_best'] = results['best_ll']
    res['BIC_best'] = results['bic']
    res['LPT_best'] = results['lpt']

    #### Simulate agent ####

    # Simulate agent for comparison of reward rate and probability of choosing best arm
    behavior_simulated = simulate_agent(
        agent_class=agent_class, 
        params=results['best_params'],
        env=env,
        fixed_params=fixed_params
    )
    reward_rate = behavior_simulated['reward'].mean()
    proba_best = (behavior_simulated['action'] == behavior_simulated['best_arm']).mean()

    #### Save results
    res['Reward rate'] = reward_rate
    res['Proba best'] = proba_best

    return pd.Series(res), behavior_simulated


# Init parameters

N_FIRST = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # number of trials at the start of each block to fit (10 or 15)

ANALYSIS_PARAMETERS = {
    "n_calls": 350,
    "n_initial_points": 100,
    "n_cpus": max(1, os.cpu_count() - 2),  
    "n_jobs": 1,                                                   
    "verbose": False,
    "make_plots": False,
    "CV_splits": None,
}

floc = os.path.join(RESULTS_PATH, 'behav_modeling', f'fitting_first{N_FIRST}')

fit_params = {
    "alpha": Real(0.01, 1, name="alpha"),
    "beta": Real(0.5, 80.0, name="beta"),
    "stickiness_bias": Real(0.0, 50.0, name="stickiness_bias"),
    "V0": Real(0.05, 0.4, name="V0"),
}

MODELS = {
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
}


# Run parameter fitting per monkey

def _run_model_fitting(model_name, behav_monkey, env):
    model_params = MODELS[model_name]

    agent_class = model_params["agent_class"]
    fixed_params = model_params["fixed_params"]
    free_params = model_params["free_params"]
    param_space = [fit_params[p] for p in free_params]

    results, behaviors_simulated = fit_simulate(
        agent_class,
        param_space,
        env,
        behav_monkey,
        fixed_params,
        fit_on_first_10=True,
        n_first=N_FIRST,
        CV_splits=ANALYSIS_PARAMETERS["CV_splits"],
        make_plots=ANALYSIS_PARAMETERS["make_plots"],
        n_calls=ANALYSIS_PARAMETERS["n_calls"],
        n_initial_points=ANALYSIS_PARAMETERS["n_initial_points"],
        n_jobs=ANALYSIS_PARAMETERS["n_jobs"],
    )
     
    # print(f'Fitted {model_name}')

    return {model_name: results}, {model_name: behaviors_simulated}


def _run_model_fitting_star(args):
    """Unpack args for Pool.imap/imap_unordered."""
    return _run_model_fitting(*args)

if __name__ == '__main__':
    for monkey in ['ka', 'po']:  # ['ka', 'po', 'yu_sham', 'yu_DCZ']
        print(f'--- {monkey} ---')

        ### Get data
        env = gym.make("zsombi/monkey-bandit-task-v0", n_arms=3, max_episode_steps=100_000)
        behav_monkey = get_data_custom(monkey)
        results = {}
        behaviors_simulated = {f'MONKEY {monkey.upper()}': behav_monkey}

        ### Fit models in parallel
        tasks = [
            (model_name, behav_monkey, env)
            for model_name in MODELS.keys()
        ]

        with Pool(ANALYSIS_PARAMETERS['n_cpus']) as pool:
            recovery_results = list(
                tqdm(
                    pool.imap_unordered(_run_model_fitting_star, tasks),
                    total=len(tasks),
                    desc=f"Fitting models ({monkey})",
                    unit="model",
                )
            )

        # Collect results
        for res_dict, behav_dict in recovery_results:
            results.update(res_dict)
            behaviors_simulated.update(behav_dict)

        # sort results[Model] MODELS.keys() (same order as in MODELS, but start with "MONKEY {upper(monkey)}")
        os.makedirs(floc, exist_ok=True)
        save_res_and_behav(results, behaviors_simulated, monkey, floc)


    print('saved all')
