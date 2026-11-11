"""
Helper functions for the Fig. 2 notebooks.
"""
import os
import numpy as np
import pandas as pd

from popy.simulation_tools import QLearner, ForagingAgent
from popy.simulation_helpers import simulate_agent
from popy.config import RESULTS_PATH


def load_optimizing_results():
    """load_res() of the original eval_model_optimizing notebook"""
    # show bic and LPT values
    
    # load behavior data
    floc = os.path.join(RESULTS_PATH, 'behav_modeling', 'optimizing', f'optimization_results.csv')
    res_temp = pd.read_csv(floc)

    # simulations
    sim_floc = os.path.join(RESULTS_PATH, 'behav_modeling', 'optimizing', f'simulation_behaviors.pkl')
    sim_data = pd.read_pickle(sim_floc)
    sim_data['best_arm'] = sim_data['action'] == sim_data['best_arm']
    for model in res_temp['Model'].unique():
        prop_best_target = sim_data.loc[sim_data['model'] == model, 'best_arm'].mean()
        res_temp.loc[res_temp['Model'] == model, 'prop_best_target'] = prop_best_target

    return res_temp


def load_fitting_results(folder=None):
    """load_res() of the original eval_model_fitting notebook"""
    # show bic and LPT values
    dfs = []
    for monkey in ['ka', 'po']:#:, 'yu_sham', 'yu_DCZ']:
        # load behavior data
        if folder is None:
            floc = os.path.join(RESULTS_PATH, 'behav_modeling', 'fitting', f'simulation_results_{monkey}.csv')
        else:
            floc = os.path.join(RESULTS_PATH, 'behav_modeling', folder, f'simulation_results_{monkey}.csv')

        res_temp = pd.read_csv(floc)

        # add monkey name to the dataframe and move to first col
        res_temp['monkey'] = monkey
        

        dfs.append(res_temp)
    return pd.concat(dfs, ignore_index=True)


# local functions
def grid_search_RL(env, n_samples=15):
    alpha_range = np.linspace(.01, .5, n_samples)  # learning rate range

    res_grid_qlearn_inferential = np.zeros(len(alpha_range))  # to store results
        
    for i, alpha in enumerate(alpha_range):
        params_temp = {'alpha': alpha}
        behavior_temp = simulate_agent(QLearner, params=params_temp, env=env, fixed_params={'structure_aware': True, 'stickiness_bias': 0})  # simulate inferential Q-learning agent (structure-aware means inferential RL setup)

        # compute proportion of best arm selections
        prop_best_arm = (behavior_temp["best_arm"].values == behavior_temp["action"].values).mean()
        res_grid_qlearn_inferential[i] = prop_best_arm 

    return alpha_range, res_grid_qlearn_inferential


def grid_search_foraging(env):
    n_samples = 15  # number of samples in the grid search (per parameter)
    alpha_range = np.linspace(.05, .8, n_samples)  # learning rate range
    V0_range = np.linspace(0, .75, n_samples)

    res_grid = np.zeros((len(alpha_range), len(V0_range)))  # to store results

    for i, alpha in enumerate(alpha_range):
        for j, V0 in enumerate(V0_range):
            params_temp = {
                'alpha': alpha,
                'V0': V0
            }
            behavior = simulate_agent(ForagingAgent, params=params_temp, env=env, fixed_params={'reset_on_switch': True})  # simulate foraging agent (reset value on switching)

            # compute proportion of best arm selection
            prop_best_arm = (behavior["best_arm"].values == behavior["action"].values).mean()
            res_grid[i, j] = prop_best_arm 

    return alpha_range, V0_range, res_grid
