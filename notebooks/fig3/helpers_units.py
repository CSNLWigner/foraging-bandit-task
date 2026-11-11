"""
Helper functions for the Fig. 3b notebook (example units).
"""
import numpy as np

from popy.io_tools import load_behavior, load_neural_data
from popy.behavior_data_tools import add_foraging_value, drop_time_fields
from popy.neural_data_tools import *


def get_data(monkey, session):
    # get behavior data
    behav = load_behavior(monkey, session)
    behav = drop_time_fields(behav)  
    behav = add_foraging_value(behav, digitize=True, n_classes=4) # add value function
    behav.dropna(subset=['stay_value'])  # drop nans in value
    
    # get neural data
    neural_data = load_neural_data(monkey, session, hz=1000)  # load neural data
    neural_data = remove_trunctuated_neurons(neural_data, mode='set_nan', delay_limit=20, print_usr_msg=True)  # set indices to nan
    neural_data = add_firing_rates(neural_data, 'gauss')  # add firing gauss
    neural_data = downsample_time(neural_data, 100)  # downsample neural data
    
    # build dataset    
    neural_dataset = build_trial_dataset(neural_data, mode='centering', center_on_epoch_start=5, center_window=(-3, 3))
    neural_dataset = merge_behavior(neural_dataset, behav)  # merge behavior and neural data

    return neural_data, neural_dataset


def _prepare_data(neural_data, coord_name):
        neural_data_temp = neural_data.copy()
        
        # get only the trials where the coord_name is not nan, and sort them
        non_nan_coord_ids = np.where(~np.isnan(neural_data_temp[coord_name].data))[0]  # get the non-nan indices
        non_nan_coord_trials = neural_data_temp.trial_id.data[non_nan_coord_ids]  # get the non-nan trials
        non_nan_coord_labels = neural_data_temp[coord_name].data[non_nan_coord_ids]  # get the non-nan labels
        non_nan_coord_trials = non_nan_coord_trials[np.argsort(non_nan_coord_labels)]  # sort the trials by the labels
        neural_data_temp = neural_data_temp.sel({'trial_id': non_nan_coord_trials})  # remove nan values (nan coordinates) and sort the trials by the labels

    
        return neural_data_temp
