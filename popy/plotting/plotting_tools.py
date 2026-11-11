"""
Ploting methods generally used ofer multiple different analysis.
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from popy.neural_data_tools import *
from popy.behavior_data_tools import *

import os
import warnings


import numpy as np
import matplotlib.pyplot as plt

from popy.config import COLORS, MODEL_PARAMS
from matplotlib.colors import to_rgba


### behavior plotting tools
##% helper functions
def get_prop_event(behav_all_pd_original, column='switch'):
    """
    Calculates the proportion of events for each monkey and session, across blocks.

    Parameters
    ----------
    behav_all_pd_original : pd.DataFrame
        The original behavior DataFrame containing all trials.
    column : str
        The column name to calculate the proportion of events for.

    Returns
    -------
    dict
        A dictionary containing the proportion of events for each monkey. For each monkey, it contains a list of arrays, where each array corresponds to a block and contains the events in the block (interpolated to a fixed length).
    """
    behav_all_pd = behav_all_pd_original.copy()

    monkeys = behav_all_pd['monkey'].unique()
    percent_selection = {monkey: [] for monkey in monkeys}

    # nominal length = 40 - blocks will be interpolated to this length
    LEN_BLOCK = 40

    # fill nans and convert to float
    behav_all_pd[column] = behav_all_pd[column].fillna(method='ffill')
    behav_all_pd[column] = behav_all_pd[column].fillna(method='bfill')
    behav_all_pd[column] = behav_all_pd[column].astype(float)  # see if its best target

    for monkey, behav_monkey in behav_all_pd.groupby('monkey'):
        for session, behav_session in behav_monkey.groupby('session'):
            for block_id, behav_block in behav_session.groupby('block_id'):
                if len(behav_block) < 35:
                    continue

                # get best target selection and shift vectors per block
                column_vector = behav_block[column]

                # interpolate to LEN_BLOCK
                column_vector = np.interp(np.linspace(0, LEN_BLOCK, LEN_BLOCK), np.linspace(0, LEN_BLOCK, len(column_vector)), column_vector)

                # append to list
                percent_selection[monkey].append(column_vector)

        # create array of best target selection
        percent_selection[monkey] = np.array(percent_selection[monkey])
        #percent_shift[monkey] = np.array(percent_shift[monkey])
        
    return percent_selection# percent_shift


def plot_strategy(behav, ax=None, saveas=None, paper_format=False, title=None, show_error=True, verbose=False, h=None, w=None):
    # init fig
    if ax is None:
        if paper_format:
            # fontsize = 18
            plt.rcParams.update({'font.size': 8})
            h_, w_ = 2*.8 + 0.15, 2*.95  # height and width in cm
            if h is not None:
                h_ = h
            if w is not None:
                w_ = w
            fig, axs = plt.subplots(2, 1, figsize=(w_, h_), sharex=True, gridspec_kw={'hspace': 0.05})  # paper format
            lw = 1.5
        else:
            plt.rcParams.update({'font.size': 12})
            fig, axs = plt.subplots(2, 1, figsize=(4, 4), sharex=True, gridspec_kw={'hspace': 0.05})
            lw = 2
    else:
        # split the provided axis into two subplots
        # Split the provided axis into 2 subplots
        fig = ax.get_figure()
        
        # Get the position of the original axis
        pos = ax.get_position()
        ax.remove()  # Remove the original axis
        
        # Create two new axes in the same space
        ax1 = fig.add_axes([pos.x0, pos.y0 + pos.height/2, pos.width, pos.height/2])  # Top
        ax2 = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height/2])  # Bottom

        axs = [ax1, ax2]

        lw = 2

    if title is not None:
        fig.suptitle(title)

    # get best target selection and shift vectors
    behav = add_switch_info(behav)
    behav['if_best_target'] = behav['target'] == behav['best_target']
    if 'model' not in behav.columns:
        behav['model'] = 'recording'

    # get percent selection and percent shift per monkey and model
    for monkey, df_monkey in behav.groupby('monkey'):
        for model, df_model in df_monkey.groupby('model'):
            # print percent best target selection for each monkey and model
            print(f"Monkey: {monkey}, Model: {model}, Percent best target selection: {df_model['if_best_target'].mean():.2f}, Percent shift: {df_model['switch'].mean():.2f}")
            
            # get percent correct and percent shift
            percent_selection = get_prop_event(df_model, 'if_best_target')
            percent_shift = get_prop_event(df_model, 'switch')
    
            LEN_BLOCK = percent_selection[list(percent_selection.keys())[0]].shape[1]

            # get mean and std for best target selection and shift
            mean_best_selection = np.mean(percent_selection[monkey], axis=0)
            std_best_selection = np.std(percent_selection[monkey], axis=0)
            sem_best_selection = std_best_selection / np.sqrt(len(percent_selection[monkey]))

            mean_shift = np.mean(percent_shift[monkey], axis=0)
            std_shift = np.std(percent_shift[monkey], axis=0)
            sem_shift = std_shift / np.sqrt(len(percent_shift[monkey]))

            # Check if model name ends with _short to use dashed line
            if monkey.endswith('_short'):
                linestyle = 'dashed'
                monkey_base = monkey[:-6]  # Remove '_short' suffix
            else:
                linestyle = 'solid'
                monkey_base = monkey
            
            if monkey_base in COLORS.keys():
                color = COLORS[monkey_base]
            else:
                color = None
                
            if linestyle == 'solid':
                if show_error and monkey in ['ka', 'po', 'yu_sham', 'yu_DCZ']:
                    axs[0].fill_between(np.arange(LEN_BLOCK), mean_best_selection-sem_best_selection, mean_best_selection+sem_best_selection, color=color, alpha=0.5)
            
            axs[0].plot(mean_best_selection, color=color, label=monkey.upper(), linestyle=linestyle, linewidth=lw, alpha=0.8)

            axs[1].plot(mean_shift, color=color , label=monkey.upper(), linestyle=linestyle, linewidth=lw, alpha=0.8)
            if linestyle == 'solid' and show_error and monkey_base in ['ka', 'po', 'yu_sham', 'yu_DCZ']:
                axs[1].fill_between(np.arange(LEN_BLOCK), mean_shift-sem_shift, mean_shift+sem_shift, color=color, alpha=0.4)

    # print stats 
    for monkey, df_monkey in behav.groupby('monkey'):
        for model, df_model in df_monkey.groupby('model'):
                
            mean_best_all = []
            for session, df_session in df_model.groupby('session'):
                percent_selection = get_prop_event(df_session, 'if_best_target')
                percent_shift = get_prop_event(df_session, 'switch')

                mean_best_temp = np.mean(percent_selection[monkey])
                mean_best_all.append(mean_best_temp)
            mean_best_all = np.array(mean_best_all)

            mean_best = np.mean(mean_best_all)
            std_best = np.std(mean_best_all)
            if verbose:
                print(f"{monkey} {model} - Best target selection: {mean_best:.2f} ± {std_best:.6f}")

    # plot settings
    ax = axs[0]
    ax.axvline(LEN_BLOCK-5, color='k', linestyle='--', alpha=0.5, zorder=-1)
    #ax.set_title('Best target selection')
    ax.set_ylabel('prob HIGH target')
    # set ticks every .2
    ax.set_yticks(np.arange(0.2, 1.1, 0.2))
    ax.set_ylim([0.2, 1])
    if not paper_format:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax = axs[1]
    ax.axvline(LEN_BLOCK-5, color='k', linestyle='--', alpha=0.5)
    #ax.set_title('Shift')
    ax.set_xlabel('Trials in block')
    ax.set_ylabel('Proba. switch')
    ax.set_yticks(np.arange(0, .8, 0.1))
    ax.set_ylim([0, .4])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, axs

def make_color_lighter(color, amount=0.1):
    """
    Makes a color lighter by a given amount.
    
    Parameters
    ----------
    color : str
        The color to make lighter, in hex format (e.g., '#90c6f2').
    amount : float, optional
        The amount to lighten the color by. Default is 0.5.
    
    Returns
    -------
    str
        The lighter color in hex format.
    """
    rgba = to_rgba(color)
    rgba = [min(1, c + amount) for c in rgba[:3]] + [rgba[3]]  # Keep alpha channel unchanged
    return to_rgba(rgba)[:3]  # Return RGB only

def show_target_selection_compact(
        session_data_original, 
        title=None, 
        background_values=[], 
        background_value_lims=None,
        ylabel=None,
        savedir=None, 
        format='paper',
        show=True):
    """
    Generates a figure illustrating the target selection, feedback, and target value.

    Parameters
    ----------
    session_data_original : pandas.DataFrame
        The original session data. 
    title : str, optional
        The title of the figure. Default is None.
    background_value : str, optional
        The name of the column in the session data that contains the value to plot in the background. Default is None.
    savedir : str, optional
        The directory to save the figure. Default is None.
    show : bool, optional
        Whether to display the figure. Default is True.

    Returns
    -------
    None
    """
    # work on a copy of the original data
    session_data = session_data_original.copy()

    if background_values is None:
        background_values = []
    
    if not isinstance(background_values, list):
        background_values = [background_values]

    if len(session_data['monkey'].unique()) != 1:
        raise ValueError('session_data should contain only one monkey')
    monkey = session_data['monkey'].unique()[0]
    #v0 = MODEL_PARAMS[monkey]['V0']

    # set the colors of the targets in RGBA format
    target_colors = {1: '#90c6f2ff', 2: '#ffb273ff', 3: '#dea8ddff'}

    # add 'trial in session' column
    if 'trial_id_in_block' not in session_data.columns:
        session_data = add_trial_in_block(session_data)  # add 'trial in block' column
    
    # init plot
    cm_to_in = 0.393701
    n_rows = len(session_data['block_id'].unique())
    if format == 'paper':
        h = 2.5  # height of each block in cm
        w = 9  # width of each block in cm
        s_marker = 16  # size of the marker
        linewidth_marker = 1.2
        height_ratios = [1.5, 3]  # for the inner grid
        hspace = 0
    elif format == 'poster':
        h = 3.5
        w = 10
        s_marker = 12
        linewidth_marker = 1
        height_ratios = [1, 5]  # for the inner grid
        hspace = 0
    else:
        h = 4
        w = 15
        s_marker = 40
        linewidth_marker = 1
        height_ratios = [1, 3]  # for the inner grid
        hspace = .1
        show_block_id = True
    fig = plt.figure(figsize=(w*cm_to_in, h*n_rows*cm_to_in))
    outer_grid = plt.GridSpec(n_rows, 1)  # Create the main grid for blocks

    for i in range(n_rows):
        # Create a subdivision of the block's grid space
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                    subplot_spec=outer_grid[i],
                                                    height_ratios=height_ratios,
                                                    hspace=hspace)
        
        # Create the two axes for this block
        ax_markers = plt.Subplot(fig, inner_grid[0])
        ax_measure = plt.Subplot(fig, inner_grid[1])
        fig.add_subplot(ax_markers)
        fig.add_subplot(ax_measure)

        # set background of axmarkers to best target color
        best_target = session_data.loc[session_data['block_id'] == i, 'best_target'].values[0]
        # Convert the color to RGBA with alpha
        rgba_color = to_rgba(target_colors[best_target], alpha=0.2)

        # Set the face color of the axis
        ax_markers.set_facecolor(rgba_color)


        # plot the target selection
        for target_id in [1, 2, 3]:  # plot the targets
            # get the selected target and its id, for REWARDED trials
            rewarded_trials_of_high_target = session_data.loc[
                (session_data['block_id'] == i) & (session_data['feedback'] == True) & (session_data['target'] == target_id) & (session_data['best_target'] == target_id)
                ]['trial_id_in_block']
            rewarded_trials_of_low_target = session_data.loc[
                (session_data['block_id'] == i) & (session_data['feedback'] == True) & (session_data['target'] == target_id) & (session_data['best_target'] != target_id)
                ]['trial_id_in_block']
            unrewarded_trials_of_high_target = session_data.loc[
                (session_data['block_id'] == i) & (session_data['feedback'] == False) & (session_data['target'] == target_id) & (session_data['best_target'] == target_id)
                ]['trial_id_in_block']
            unrewarded_trials_of_low_target = session_data.loc[
                (session_data['block_id'] == i) & (session_data['feedback'] == False) & (session_data['target'] == target_id) & (session_data['best_target'] != target_id)
                ]['trial_id_in_block']
            
            ax_markers.scatter(
                rewarded_trials_of_high_target,
                np.ones_like(rewarded_trials_of_high_target) * target_id,
                color=target_colors[target_id], marker='o', facecolors=make_color_lighter(target_colors[target_id]), s=s_marker, linewidth=linewidth_marker)  # plot rewarded trials
            ax_markers.scatter(
                rewarded_trials_of_low_target,
                np.ones_like(rewarded_trials_of_low_target) * target_id,
                color=target_colors[target_id], marker='o', facecolors=make_color_lighter(target_colors[target_id]), s=s_marker, linewidth=linewidth_marker)  # plot rewarded trials
            ax_markers.scatter(
                unrewarded_trials_of_high_target,
                np.ones_like(unrewarded_trials_of_high_target) * target_id,
                color=target_colors[target_id], marker='o', facecolors='white', s=s_marker, linewidths=linewidth_marker)
            ax_markers.scatter(
                unrewarded_trials_of_low_target,
                np.ones_like(unrewarded_trials_of_low_target) * target_id,
                color=target_colors[target_id], marker='o', facecolors='white', s=s_marker, linewidths=linewidth_marker)

        # plot interrupted trials
        interrupted_trials_id = session_data.loc[
            (session_data['block_id'] == i) & (session_data['feedback'].isnull())
            ]['trial_id_in_block']

        if len(interrupted_trials_id) > 0:
            ax_markers.scatter(
                interrupted_trials_id,
                np.ones_like(interrupted_trials_id),
                color='black', marker='x', label='interrupted trial')
       
        '''# plot green lines marking the best target
        ax.plot(np.arange(0, no_of_trials, 1),
                np.ones((no_of_trials,)) * session_data.loc[session_data['block_id'] == i, 'best_target'],
                label='best target', color='green', alpha=.3, linewidth=20)'''

        # plot MEASURE (i.e. value)
        bg_val_colors = {'stay_value': '#a02c2cff', 'V0': '#a08a2cff', 
                         'Q_1': target_colors[1], 'Q_2': target_colors[2], 'Q_3': target_colors[3]}
        alphas = {'V0': 0.5}
        if background_values is not None:
            for background_value in background_values:
                measure = session_data.loc[(session_data['block_id'] == i), background_value].to_numpy()
                color = bg_val_colors.get(background_value, 'grey')
                alpha = alphas.get(background_value, .7)
                ax_measure.plot(measure, color=color, alpha=alpha)  # plot the measure on the twin axis

        # mark last 5 trials per block (here the reward probabilities gradually change)
        no_of_trials = len(session_data.loc[session_data['block_id'] == i])  # number of trials in the block
        for ax in [ax_markers, ax_measure]:
            ax.axvline(no_of_trials - 6 + 0.5, linestyle='dashed', 
                        color='black', alpha=0.5)

        ## PLOT SETTINGS
        #ax_markers.title.set_text('block id: ' + str(i))
        ax_markers.set_title('block id: ' + str(i))

        ax_markers.set_xlim(-0.5, 45)
        ax_markers.set_ylim(0., 4)
        ax_markers.set_xticks(range(0, 45, 10))
        ax_markers.set_xticklabels([])
        ax_markers.set_yticks([])
        ax_markers.set_ylabel('')

        ax_measure.set_xlim(-0.5, 45)
        if background_value_lims is None:
            ax_measure.set_ylim(-.05, 1.05)
            ax_measure.axhline(0, color='grey', linewidth=0.5, linestyle='dashed')
            ax_measure.axhline(1, color='grey', linewidth=0.5, linestyle='dashed')
        else:
            ax_measure.set_ylim(background_value_lims)
            ax_measure.axhline(0, color='grey', linewidth=0.5, linestyle='-')

        if i == n_rows - 1:
            ax_measure.set_xlabel('trials in block (trials in session)')
        if background_value is not None:
            ax_measure.set_ylabel(ylabel if ylabel is not None else background_value)

        # set xticks to trial ids
        trial_ids = session_data.loc[session_data['block_id'] == i, 'trial_id'].values
        xticks = []
        for x in range(0, 45, 10):
            if x < len(trial_ids):
                xticks.append(f'{str(x+1)} ({str(trial_ids[x]+1)})')
            else:
                xticks.append(str(x+1))
        ax_measure.set_xticks(range(0, 45, 10))
        ax_measure.set_xticklabels(xticks)

        # remove spines
        ax_markers.spines['top'].set_visible(False)
        ax_markers.spines['right'].set_visible(False)
        ax_markers.spines['bottom'].set_visible(False)

        ax_measure.spines['top'].set_visible(False)
        ax_measure.spines['right'].set_visible(False)
        ax_measure.spines['bottom'].set_visible(False)

        '''ax_measure.grid()
        ax_markers.grid()'''

    # add legend to last subplot, below in the middle
    ax_markers.scatter([], [], color='grey', marker='o', facecolors='grey', s=30, label='Rewarded ')
    ax_markers.scatter([], [], color='grey', marker='o', facecolors='none', s=30, label='Non-Rewarded ')
    ax_markers.scatter([], [], color=target_colors[1], marker='o', label='target 1')   
    ax_markers.scatter([], [], color=target_colors[2], marker='o', label='target 2')
    ax_markers.scatter([], [], color=target_colors[3], marker='o', label='target 3')
    ax_markers.legend(loc='center', bbox_to_anchor=(.5, -8), ncol=1)

    plt.tight_layout()
    if title is not None:
        plt.suptitle(title, y=1)

    if savedir is not None:
        plt.savefig(savedir, dpi=300, bbox_inches='tight', transparent=False)
    if show:
        plt.show()
    else:
        plt.close()


def _get_ticks(n_extra_trials):
    n, m = n_extra_trials

    # Base ticks and labels for the current trial
    base_ticks = [0, 1, 2, 2.5, 3, 3.5, 7.5]
    base_labels = ['st', 'Lt', 'Lv', 'Tt', 'Tv', 'Fb']

    # Generate ticks and labels for extra trials
    xticks = []
    xticklabels = []

    # Add ticks and labels for past trials
    for i in range(n, 0):
        offset = i * (base_ticks[-1])
        xticks.extend([offset + tick for tick in base_ticks[:-1]])
        xticklabels.extend([f'${label} (t{i})$' for label in base_labels])

    # Add ticks and labels for the current trial
    xticks.extend(base_ticks[:-1])
    xticklabels.extend([f'${label} (t)$' for label in base_labels])

    # Add ticks and labels for future trials
    for i in range(1, m + 1):
        offset = i * (base_ticks[-1])
        xticks.extend([tick + offset for tick in base_ticks[:-1]])
        xticklabels.extend([f'${label} (t+{i})$' for label in base_labels])

    # Add the final tick and label for the last future trial
    xticks.append(base_ticks[-1] + m * base_ticks[-1])
    xticklabels.append(f'$fin (t+{m})$' if m > 0 else '$fin (t)$')
    
    return xticks, xticklabels
    
    
def plot_keypoints(ax=None, n_extra_trials=(0, 0), n_trials=None, mark_event='none', fontsize=None, axis='x', rotation=90, xlabels='both'):
    '''
    Plot the key points of the task on the x-axis of the plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot the key points on.
    n_extra_trials : int, optional
        The number of extra trials to plot, can be either positiv or negativ or 0. The default is 1.
    n_trials : int, optional
        Deprecated. Use n_extra_trials instead.
    fontsize : int, optional
        The fontsize of the xticks. The default is None.
    axis : str, optional
        The axis to plot the key points on, can be either 'x', 'y' or 'both' (both for plotting e.g. cross-correlation matrices). The default is 'x'.
    rotation : int, optional
        The rotation of the xticks. The default is 90.
    xlabels : str, optional
        Weather to plot the xlabels as events or timepoints. Can be 'events' or 'time' or 'both'. The default is 'events'.
    
    '''    

    assert n_trials is None, "n_trials is deprecated. Use n_extra_trials instead."
    assert isinstance(n_extra_trials, tuple), "n_extra_trials must be a tuple."    
    
    if ax is None:
        fig, ax = plt.subplots()    
    
    xticks, xticklabels = _get_ticks(n_extra_trials)

    # it is possible toa dd the keypoints on the y axis as well (e.g. cross correlation matrices, etc..)
    if axis == 'x' or axis == 'both':
        ax.set_xticks(xticks)
        if xlabels == 'events':
            ax.set_xticklabels(xticklabels, fontsize=fontsize, rotation=rotation)
        elif xlabels == 'time':
            ax.set_xticklabels([str(tick_) for tick_ in xticks], fontsize=fontsize, rotation=rotation)
            ax.set_xlabel('time (s)')
        elif xlabels == 'both':
            x_ticks_combined = [f"{label_} | {tick_}s" for label_, tick_ in zip(xticklabels, xticks)]
            ax.set_xticklabels(x_ticks_combined, fontsize=fontsize, rotation=rotation)
            ax.set_xlabel('time (s)')
        elif xlabels == 'short':
            ax.set_xticklabels([label.split('$')[1].split('(')[0] for label in xticklabels], fontsize=fontsize, rotation=rotation)

    if axis == 'y' or axis == 'both':
        ax.set_yticks(xticks)
        if xlabels == 'events':
            ax.set_yticklabels(xticklabels, fontsize=fontsize, rotation=rotation)
        elif xlabels == 'time':
            ax.set_yticklabels([str(tick_) for tick_ in xticks], fontsize=fontsize, rotation=rotation)
            ax.set_ylabel('time (s)')
        elif xlabels == 'both':
            y_ticks_combined = [f"{label_} | {tick_}s" for label_, tick_ in zip(xticklabels, xticks)]
            ax.set_yticklabels(y_ticks_combined, fontsize=fontsize, rotation=90-rotation)
            ax.set_ylabel('time (s)')
        elif xlabels == 'short':
            ax.set_yticklabels([label_.split('$')[1].split('(')[0] for label_ in xticklabels], fontsize=fontsize, rotation=rotation)

    if mark_event != 'none':
        for xtick_temp, xtick_label_temp in zip(xticks, xticklabels):
            if xtick_label_temp.replace("$", "").split(' ')[0] == mark_event:
                if axis == 'x' or axis == 'both':
                    ax.axvline(xtick_temp, color='black')
                if axis == 'y' or axis == 'both':
                    ax.axhline(xtick_temp, color='black')
    return ax


def plot_glm_weights(df_res, fit_curve=False, title=None, saveas=None, paper_format=False):
    # Plot the results
    # 18 pt font size
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h, w = 1.75*.8, 2
        s = 30  # marker size
        lw = 1
    else:
        plt.rcParams.update({'font.size': 18})
        cm = 1/2.54  # centimeters in inches
        h, w = 4, 6 
        s = 100  # marker size
        lw = 5  # line width

    fig, ax = plt.subplots()
    fig.set_size_inches(w, h)

    if title is not None:
        plt.suptitle(title)

    # scale weights for both monkeys by dividing with the maximum absolute value
    #df_res['coeffs'] = df_res['coeffs'] / df_res.groupby('monkey')['coeffs'].transform('max')

    legend_elements = []  # Store legend elements for each monkey

    # Create scatter plot for each monkey
    m_count = 0
    for monkey, group in df_res.groupby('monkey'):
        # Convert variable names to numeric x-positions
        x_positions = [list(df_res['variable'].unique()).index(var) + m_count * 0.15 for var in group['variable']]
        m_count += 1  # increment monkey count for the next monkey

        # Get color from COLORS dictionary using monkey key
        current_color = COLORS[monkey]
        linestyle = 'dashdot' if monkey.split('_')[0] == 'simulation' else 'solid'
        alpha = 0.5 if monkey.split('_')[0] == 'simulation' else 1
        
        # Plot points with different fill styles based on significance
        pval_col = 'pvalue_fdr' if 'pvalue_fdr' in group.columns else 'pvalue'
        print(f'Using p-value column: {pval_col}')
        significant = group[pval_col] < 0.05

        # fit exponential curve of c(a(1-a)^t)
        if fit_curve:
            y = group['coeffs']
            popt, pcov = curve_fit(lambda x, c, a: c * (a * (1 - a) ** x), x_positions, y)
            print(f'{monkey}: c={popt[0]:.2f}, a={popt[1]:.2f}')
            ax.plot(x_positions, popt[0] * (popt[1] * (1 - popt[1]) ** x_positions), color='red', linestyle='--', alpha=0.5, linewidth=lw)
            print(f'{monkey}: c={popt[0]:.2f}, a={popt[1]:.2f}')
            
        # Plot significant points (filled)
        ax.scatter(
            [x for x, sig in zip(x_positions, significant) if sig],
            [y for y, sig in zip(group['coeffs'], significant) if sig],
            color=current_color,
            s=s,
            marker='o',
            label=f'{monkey.upper()}, alpha={popt[1]:.2f}' if fit_curve else f'{monkey.upper()}',
            alpha=alpha,
        )
        
        # Plot non-significant points (empty)
        ax.scatter(
            [x for x, sig in zip(x_positions, significant) if not sig],
            [y for y, sig in zip(group['coeffs'], significant) if not sig],
            color=current_color,
            s=s,
            marker='o',
            facecolors='none',
            alpha=alpha
        )

        '''# plot stds
        ax.errorbar(
            x_positions,
            group['sampled_coeffs_mean'],
            yerr=group['sampled_coeffs_std'],
            fmt='o',
            color=current_color,
            alpha=0.5,
            capsize=3,
            elinewidth=1,
            capthick=1
        )'''
        
        # Add lines connecting points
        ax.plot(x_positions, group['coeffs'], color='grey', alpha=.6, linestyle=linestyle, zorder=0)

        '''# Create legend elements for this monkey
        legend_elements.extend([
            plt.scatter([], [], color=current_color, s=30, marker='o', label=f'{monkey.upper()}'),
        ])'''

    ax.axhline(0, color='black', linewidth=lw*.5, linestyle='dashed', alpha=0.6)

    # Add significance markers to legend
    plt.scatter([], [], color='gray', s=s, marker='o', label='p < 0.05'),
    plt.scatter([], [], color='gray', s=s, marker='o', facecolors='none', label='n.s.')

    # add zero baseline
    #ax.axhline(0, color='black', linewidth=0.5)
    #ax.set_ylim(0, 1.1)

    # Plot settings
    #ax.set_ylabel('Reward history weight')
    ax.set_ylabel('Regression coefficient')
    
    ax.set_xlabel('Trials in past')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)

    # Set x-ticks with t-n format
    ax.set_xticks(range(len(df_res['variable'].unique())))
    ax.set_xticklabels([f't-{i+1}' for i in range(len(df_res['variable'].unique()))], rotation=45)

    # Add legend with both monkey names and significance indicators
    ax.legend(frameon=False)

    plt.tight_layout()
    
    if saveas is not None:
        plt.savefig(saveas, dpi=300, bbox_inches='tight', transparent=True)
