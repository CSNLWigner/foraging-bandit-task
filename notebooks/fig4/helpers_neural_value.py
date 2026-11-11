##% title Imports
import os
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV, Lasso
from scipy import stats
import matplotlib.colors as mcolors
from scipy import ndimage
from joblib import Parallel, delayed
import statsmodels.api as sm

from matplotlib.collections import LineCollection

from collections import defaultdict


from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.neural_data_tools import time_normalize_session, scale_neural_data, remove_low_fr_neurons, remove_trunctuated_neurons
from popy.decoding.decoder_tools import *
from popy.plotting.plotting_tools import *
#from popy.plotting.plot_behavior import show_target_selection 
import popy.config as cfg
from popy.config import value_gradient
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import axes
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

PATH = cfg.PROJECT_PATH_LOCAL

##% data loading
def load_data_custom(monkey, session, area=None, subregion=None, n_extra_trials=(0, 1), sr=100):
    # load data (meta session)
    behav = load_behavior(monkey, session)
    behav = drop_time_fields(behav)
    behav = add_switch_info(behav)
    behav = add_foraging_value(behav)
    behav = add_history_of_feedback(behav, num_trials=8, one_column=False, add_history_of_targets=False)
    behav['fb_sequence'] = [r3 + 2*r2 + 4*r1 for (r1, r2, r3) in zip(behav['R_1'], behav['R_2'], behav['R_3'])]
    behav['target_1'] = behav['target'] == 1
    behav = behav.dropna()
    
    neural_data = load_neural_data(monkey, session, hz=1000)
    neural_data = remove_low_fr_neurons(neural_data, 1, print_usr_msg=False)
    neural_data = remove_trunctuated_neurons(neural_data, mode='remove', delay_limit=10, print_usr_msg=False)
    neural_data = add_firing_rates(neural_data, drop_spike_trains=True, method='gauss', std=.05)
    neural_data = downsample_time(neural_data, sr)
    neural_data = scale_neural_data(neural_data)

    # 3. build neural dataset and merge with behavior
    neural_data = time_normalize_session(neural_data)
    neural_dataset = build_trial_dataset(neural_data, mode='full_trial', n_extra_trials=n_extra_trials)
    neural_dataset = merge_behavior(neural_dataset, behav)

    return neural_dataset


def process_fold(fold_data):
    """Process a single fold - to be run in parallel"""
    neural_dataset, fold_id, train_idx, test_idx, target, across_time = fold_data

    # Initialize models for this fold
    if len(np.unique(neural_dataset[target].values)) > 2:
        continuous = True
        clf = LogisticRegressionCV(cv=5, max_iter=1000)
    else:
        continuous = False
        clf = LogisticRegression()
        rus = RandomUnderSampler(random_state=42)
    
    neural_dataset_train = neural_dataset.sel(trial_id=neural_dataset.trial_id.values[train_idx])
    neural_dataset_test = neural_dataset.sel(trial_id=neural_dataset.trial_id.values[test_idx])

    # get labels of this fold
    y_train = neural_dataset_train[target].values
    y_test = neural_dataset_test[target].values

    n_timebins = len(neural_dataset.time.values)
    scores_fold = np.full(n_timebins, np.nan)
    across_time_scores_fold = np.full((n_timebins, n_timebins), np.nan)
    projections_fold = []
    
    for train_t_id, t in enumerate(neural_dataset.time.values):
        # get the firing rates for the current time point
        X_train_temp = neural_dataset_train.firing_rates.sel(time=t).values
        X_test_temp = neural_dataset_test.firing_rates.sel(time=t).values

        # balance dataset
        if not continuous:
            X_train_temp, y_train_temp = rus.fit_resample(X_train_temp, y_train)

        # fit the model on train data
        clf.fit(X_train_temp, y_train_temp)

        # evaluate on test set
        score = clf.score(X_test_temp, y_test)

        # project test set 
        data_projected = clf.intercept_ + X_test_temp @ clf.coef_.ravel()

        # Store score and projections
        scores_fold[train_t_id] = score
        projections_fold.append(data_projected)

        # Test this decoder across all time points
        if across_time:
            for test_t_id, test_t in enumerate(neural_dataset.time.values):
                X_test_all_times = neural_dataset_test.firing_rates.sel(time=test_t).values
                score_across_time = clf.score(X_test_all_times, y_test)
                across_time_scores_fold[train_t_id, test_t_id] = score_across_time

    # Create a DataArray with the projected data (to preserve trials)
    projections_fold = np.array(projections_fold)
    time_coords = {name: coord for name, coord in neural_dataset.coords.items() if 'time' in coord.dims}
    trial_coords_fold = {name: coord for name, coord in neural_dataset_test.coords.items() if 'trial_id' in coord.dims}
    projections_fold = xr.DataArray(projections_fold.T, dims=('trial_id', 'time'), 
                                   coords={**trial_coords_fold, **time_coords})
    
    return fold_id, scores_fold, across_time_scores_fold, projections_fold


def time_resolved_decoder_all_time(neural_dataset, target='R_1', group=None, across_time=False, n_jobs=-1):
    trial_ids = neural_dataset.trial_id.values
    labels = neural_dataset[target].values  
    cv = 10

    # create train and test splits
    if len(np.unique(neural_dataset[target].values)) > 2:
        continuous = True
        skf = KFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        continuous = False
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    n_timebins = len(neural_dataset.time.values)
    time_coords = {name: coord for name, coord in neural_dataset.coords.items() if 'time' in coord.dims}

    # Prepare data for parallel processing
    fold_data = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(trial_ids, labels)):
        fold_data.append((neural_dataset.copy('deep'), fold_id, train_idx, test_idx, target, across_time))

    # Process folds in parallel - xarray can not be pickled...
    '''print(f"Processing {cv} folds in parallel with {n_jobs} jobs...")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_fold)(data) for data in fold_data
    )'''
    results = []
    for data in fold_data:
        results.append(process_fold(data))

    # Collect results
    scores_all = np.full((n_timebins, cv), np.nan)  # (timebins, folds)
    across_time_scores = np.full((n_timebins, n_timebins, cv), np.nan)  # (folds, timebins)
    projected_data = []

    for fold_id, scores_fold, across_time_scores_fold, projections_fold in results:
        scores_all[:, fold_id] = scores_fold
        if across_time:
            across_time_scores[:, :, fold_id] = across_time_scores_fold
        projected_data.append(projections_fold)

    # Concatenate projected data
    projected_data = xr.concat(projected_data, dim='trial_id')
    projected_data = projected_data.sortby('trial_id')

    # average over folds
    #coeffs_all = np.mean(coeffs_all, axis=0)
    scores_all = np.mean(scores_all, axis=-1)  # average over folds
    across_time_scores = np.mean(across_time_scores, axis=-1)  # average over folds

    # create an xarray
    projected_data = projected_data.assign_coords(decodability=('time', scores_all))
    
    # NEW: Create xarray for across-time decodability matrix
    decodability_matrix = xr.DataArray(
        across_time_scores,
        dims=('train_time', 'test_time'),
        coords={'train_time': neural_dataset.time.values, 'test_time': neural_dataset.time.values}
    )

    return decodability_matrix, projected_data


def add_neural_value_coord(data_projected, t_interest=[2.5, 3.5]):
    trial_ids = data_projected.trial_id.values
    V_ts = []
    V_t_p1s = []
    dVs = []

    V_ts_behav = []
    V_t_p1s_behav = []
    dVs_behav = []
    for trial_id in trial_ids:
        V_t = data_projected.sel(trial_id=trial_id, time=slice(t_interest[0], t_interest[1])).mean('time').values
        V_t_behav = data_projected.sel(trial_id=trial_id).stay_value.values

        # if next trial is part of the same session, get the next time point
        if trial_id + 1 in data_projected.trial_id.values:
            V_t_p1 = data_projected.sel(trial_id=trial_id + 1, time=slice(t_interest[0], t_interest[1])).mean('time').values
            dV = V_t_p1 - V_t

            V_t_p1_behav = data_projected.sel(trial_id=trial_id + 1).stay_value.values
            dV_behav = V_t_p1_behav - V_t_behav
        else:
            # if next trial is not part of the same session, set dV to NaN
            V_t_p1 = np.nan
            dV = np.nan

            V_t_p1_behav = np.nan
            dV_behav = np.nan


        V_ts.append(V_t)
        V_t_p1s.append(V_t_p1)
        dVs.append(dV)

        V_ts_behav.append(V_t_behav)
        V_t_p1s_behav.append(V_t_p1_behav)
        dVs_behav.append(dV_behav)

    # convert to numpy arrays
    V_ts = np.array(V_ts)   
    V_t_p1s = np.array(V_t_p1s)
    dVs = np.array(dVs)

    V_ts_behav = np.array(V_ts_behav)
    V_t_p1s_behav = np.array(V_t_p1s_behav)
    dVs_behav = np.array(dVs_behav)

    data_projected = data_projected.assign_coords(V_t=('trial_id', V_ts))
    data_projected = data_projected.assign_coords(V_t_p1=('trial_id', V_t_p1s))
    data_projected = data_projected.assign_coords(dV=('trial_id', dVs))
    
    data_projected = data_projected.assign_coords(V_t_behav=('trial_id', V_ts_behav))
    data_projected = data_projected.assign_coords(V_t_p1_behav=('trial_id', V_t_p1s_behav))
    data_projected = data_projected.assign_coords(dV_behav=('trial_id', dVs_behav))

    return data_projected


def plot_projected_data(data_projected, t_interest_value=None, normalize=False, n_extra_trials=(0, 1), xlim=None, paper_format=False, ax=None):
    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 1.7  # in cm
        w = 2.5
        linewidth = 2
    else:
        plt.rcParams.update({'font.size': 12})
        h = 6  # in cm
        w = 8  # in cm
        linewidth = 1.5

    unique_fb_sequences = np.sort(np.unique(data_projected.fb_sequence.data))

    colors = value_gradient(len(unique_fb_sequences))

    if ax is None:
        fig, ax = plt.subplots(figsize=(w, h))

    # 500 ms for convolution
    dt = data_projected.time.values[1] - data_projected.time.values[0]
    n_convolution_bins = int(0.5 / dt)
    window = np.ones(n_convolution_bins) / n_convolution_bins

    time_vector = data_projected.time.values


    # bwr colormap, n=8 sampples RdYlGn
    alphas = [1-i for i in np.linspace(0, 1, len(unique_fb_sequences))]
    labels = ["[-, -, -]", "[+, -, -]", "[-, +, -]", "[+, +, -]", "[-, -, +]", "[+, -, +]", "[-, +, +]", "[+, +, +]"]
        
    for i, label in enumerate(unique_fb_sequences):
        class_mean = np.mean(data_projected.where(data_projected.fb_sequence == label), axis=0)
        class_mean_smoothed = ndimage.convolve(class_mean, window, mode='nearest')
        ax.plot(time_vector, class_mean_smoothed, color=colors[i], label=labels[i], linewidth=linewidth, zorder=10)

    if t_interest_value is not None:
        ax.axvspan(t_interest_value[0], t_interest_value[1], color='grey', alpha=0.2, label='eval window')

    # add behav keyboints
    plot_keypoints(ax, n_extra_trials, mark_event='Fb', xlabels='events')
    # y grid only
    ax.grid(axis='x', alpha=.5, linewidth=1, linestyle='--')
    ax.axhline(0, color='k', linestyle='-', linewidth=linewidth*.75, zorder=0)

    # remove left and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # move legend outside
    if not paper_format:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # save as svg
    ax.set_ylabel('$V_t$', fontsize=10)
    '''if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Neural value')'''
    
    if xlim is not None:    
        ax.set_xlim(xlim)

    return fig, ax


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
