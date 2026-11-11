"""
These functions are used to load the raw data (behavior and neural) from the folder data/recordings and to preprocess the data.

Output functions should also go here.
"""
import os
import scipy.io

import popy.config as config # project config file
from popy.io_tools_helpers import *  # behavior preprocessing functions

def load_behavior(monkey=None, session=None):
    """
    Simply loads the full behavior dataset so that its not necessary to load it every time from path.

    Monkeys 'ka' and 'po' are loaded from behavior_kapo.pkl, monkey 'yu' from behavior_yu.pkl (made by
    notebooks/preprocessing_sessions/preproc_behavioral_data/preproc_behavioral_data_yuri.py).
    Without a monkey, only 'ka' and 'po' are returned.
    """

    base_path = config.DATA_PATH
    fname = 'behavior_yu.pkl' if monkey == 'yu' else 'behavior_kapo.pkl'
    data_path = os.path.join(base_path, 'behavior', fname)
    concat_data = pd.read_pickle(data_path)

    # sort by [monkey, session, trial_id]
    concat_data = concat_data.sort_values(by=['monkey', 'session', 'trial_id'])
    
    # set 'feedback' to float
    concat_data['feedback'] = concat_data['feedback'].astype(float)
    
    if (monkey is not None) and (session is not None):
        assert session in concat_data.loc[concat_data['monkey'] == monkey, 'session'].values, f"Session {session} not found for monkey {monkey}"
        return concat_data.loc[(concat_data['monkey'] == monkey) & (concat_data['session'] == session)]
    elif (monkey is not None) and (session is None):
        return concat_data.loc[(concat_data['monkey'] == monkey)]
    elif (monkey is None) and (session is not None):
        raise ValueError("provide monkey name if you want to filter by session")    
    elif (monkey is None) and (session is None):
        return concat_data
    
def load_behavior_yuri():
    """
    Loads the behavior dataset in xarray format.
    """

    base_path = config.DATA_PATH
    data_path = os.path.join(base_path, 'processed', 'behavior', 'behavior_yuri.pkl')
    data = pd.read_pickle(data_path)

    # sort by [monkey, session, trial_id]
    return data    

def load_neural_data(monkey, session, hz=100, *args, **kwargs):
    """
    Loading function for one session's neural data in xarray format.

    Returns the spike data in xarray.Dataset format.

    Spike trains, as xarray.DataArray, are stored in the 'spike_trains' variable of the dataset and can be accessed as:
    `neural_data['spike_trains']`.

    Parameters:
        monkey (str): 'ka' or 'po'
        session (str): Session number

    Returns:
        xarray.Dataset: The neural data in xarray.Dataset format
    """
    import warnings

    if hz == 100:
        raise ValueError("100 Hz data is not supported anymore, use 1000 Hz data and downsample after firing rate generation with: .")

    # SANITY: check if there is anything in args and kwargs -> older versions of the function had 'mode' parameter and 'return_dataset_format' parameter
    if len(args) > 0:
        raise ValueError(f"'mode' and 'return_dataset_format' parameters are deprecated, use 'neural_data = load_neural_data(monkey, session)' instead, that will return xarray.Dataset.\n\
            To get xarray.DataArray of spikes, use the 'spike_trains' variable of the dataset as 'neural_data.spike_trains'.\n\
            To get firing rates, use the 'neural_data = add_firing_rates(neural_data)' function from popy.neural_data_tools, then you can access firing rates as 'neural_data.firing_rates'.")
    if len(kwargs) > 0:
        if 'mode' in kwargs:
            kwargs.pop('mode')
            raise ValueError("mode parameter is deprecated, instead of loading rates, generate it from the spike trains -> Load spike trains without ```mode``` param, \n\
                and then generate rates with 'add_firing_rates()' function from popy.neural_data_tools.")
        if 'return_dataset_format' in kwargs:
            kwargs.pop('return_dataset_format')
            raise ValueError("old convention is depricated, function will return xarray.Dataset. To get xarray.DataArray of spikes, use the 'spike_trains' variable of the dataset as 'neural_data.spike_trains'.\n\
                          To get firing rates, use the 'neural_data = add_firing_rates(neural_data)' function from popy.neural_data_tools, then you can access firing rates as 'neural_data.firing_rates'.")
        if len(kwargs) > 0:
            raise ValueError(f"Unknown parameters: {kwargs.keys()}")
        
    # Load neural data
    '''out_path = os.path.join(config.DATA_PATH, 'processed', 'neural_data')
    floc = os.path.join(out_path, f'{monkey}_{session}_spikes.nc')'''
    if hz == 1000:
        out_path = os.path.join(config.DATA_PATH, 'neural_data')

    elif hz == 100:    
        out_path = os.path.join(config.DATA_PATH, 'processed', 'neural_data_old_100hz', 'spikes')
    floc = os.path.join(out_path, f'{monkey}_{session}_spikes.nc')

    neural_data = xr.open_dataarray(floc)
    neural_data = neural_data.astype(float)
    neural_data.attrs['bin_size'] = round(neural_data.time[1].values - neural_data.time[0].values, 4)

    # make it a dataset
    neural_data = neural_data.to_dataset(name='spike_trains')
    neural_data.attrs['bin_size'] = round(neural_data.time[1].values - neural_data.time[0].values, 4)

    # Load the data into memory
    neural_data.load()

    return neural_data


def load_metadata():
    # load metadata
    metadata = pd.read_csv(os.path.join(config.DATA_PATH, 'behavior', 'recordings_summary.csv'), dtype={'session': str})
    return metadata
