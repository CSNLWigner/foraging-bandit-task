##% title Imports
import os
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV, Lasso
from scipy import stats
import matplotlib.colors as mcolors
from scipy import ndimage
from joblib import Parallel, delayed
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from scipy import stats

from matplotlib.collections import LineCollection

from collections import defaultdict


from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.neural_data_tools import time_normalize_session, scale_neural_data, remove_low_fr_neurons, remove_trunctuated_neurons
from popy.decoding.decoder_tools import *
from popy.plotting.plotting_tools import *
# from popy.plotting.plot_behavior import show_target_selection
import popy.config as cfg
from popy.config import COLORS
from popy.config import value_gradient
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GroupKFold
from matplotlib import axes
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

PATH = cfg.PROJECT_PATH_LOCAL

##% data loading
def load_data_custom(monkey, session, area=None, subregion=None, n_extra_trials=(0, 1), sr=100, n_first_only=None):
    # 1. Behavior data

    # load data (meta session)
    behav = load_behavior(monkey, session)
    behav = drop_time_fields(behav)
    
    # add behav vars to decode
    #behav = add_foraging_value(behav)
    behav = add_switch_info(behav)
    behav = add_history_of_feedback(behav, num_trials=8, one_column=False, add_history_of_targets=False)
    #behav['fb_sequence'] = [r3 + 2*r2 + 4*r1 for (r1, r2, r3) in zip(behav['R_1'], behav['R_2'], behav['R_3'])]
    #behav['target_1'] = behav['target'] == 1
    behav = behav.dropna()

    if n_first_only is not None:
        # group by monkey, session, block, and take only the first n_first_only trials of each block
        behav = behav.groupby(['monkey', 'session', 'block_id']).head(n_first_only).reset_index(drop=True)

    # 2. Neural data
    
    # Load neural data
    neural_data = load_neural_data(monkey, session, hz=1000)
    
    if area is not None:
        # Filter neural data by area
        neural_data = neural_data.sel(unit=neural_data.unit.str.contains(area))

    # remove some units
    neural_data = remove_low_fr_neurons(neural_data, 1, print_usr_msg=False)
    neural_data = remove_trunctuated_neurons(neural_data, mode='remove', delay_limit=10, print_usr_msg=False)
    
    # process neural data
    neural_data = add_firing_rates(neural_data, drop_spike_trains=True, method='gauss', std=.05)
    neural_data = downsample_time(neural_data, sr)
    neural_data = scale_neural_data(neural_data)
    neural_data = time_normalize_session(neural_data)

    # 3. build neural dataset and merge with behavior
    neural_dataset = build_trial_dataset(neural_data, mode='full_trial', n_extra_trials=n_extra_trials)
    neural_dataset = merge_behavior(neural_dataset, behav)

    return neural_dataset


### Save figures
from matplotlib.backends.backend_pdf import PdfPages
import io
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

