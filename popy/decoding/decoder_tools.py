import pandas as pd
import numpy as np

from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.neural_data_tools import time_normalize_session, scale_neural_data, remove_low_fr_neurons, remove_trunctuated_neurons
from popy.decoding.population_decoders import *
from popy.plotting.plotting_tools import *
import popy.config as cfg

from matplotlib.patches import Rectangle, FancyBboxPatch


## Data processings

def mask_N_consecutives(data, N=4, non_signif_coding='nan'):
    if non_signif_coding == 'nan':
        def criteria(num):
            return num == num
    elif non_signif_coding == '0':
        def criteria(num):
            return num != 0

    # find where N consecutive values are False
    mask = np.empty(data.shape)
    for u, unit_data in enumerate(data.values):
        consecutive_count = 0
        candidate_ids = []
        unit_signif_ids = []
        for i, num in enumerate(unit_data):
            # if not nan
            if criteria(num):
                consecutive_count += 1
                candidate_ids.append(i)  # add to candidate indices (may or may not be part of N consecutive bins)
                if consecutive_count >= N:
                    unit_signif_ids += candidate_ids  # add to list of significant indices
                    candidate_ids = []  # reset candidate ids
            else:  # reset
                consecutive_count = 0
                candidate_ids = []
        # create mask for this unit 
        unit_signif = np.zeros_like(unit_data)
        unit_signif[unit_signif_ids] = 1

        mask[u] = unit_signif

    #data_masked = data.where(mask == 1)
    if non_signif_coding == 'nan':
        return data.where(mask == 1)
    elif non_signif_coding == '0':
        return data * mask

