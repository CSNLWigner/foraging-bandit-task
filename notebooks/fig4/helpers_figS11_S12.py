"""
Helper functions for Supplementary Fig. S11 (best-fitting learning rate per session on the cortical grid).
"""
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import popy.config as cfg
from popy.plotting.plot_cortical_grid import _get_grid_location, _init_location_grid


##% data tools
def add_cortical_position(res):
    from popy.plotting.plot_cortical_grid import _get_grid_location
    """ adds xy coordinates of coortical position for each recording. """
    res = res.copy()

    # add empty coordinates over session and area (nans)
    shape = (len(res.session.data), len(res.area.data))
    array_empty = np.ones(shape = (len(res.session.data), len(res.area.data))) * np.nan
    res = res.assign_coords(x=(('session', 'area'), array_empty.copy()))
    res = res.assign_coords(y=(('session', 'area'), array_empty.copy()))
        
    # load positions
    for session in res.session.data:
        monkey_temp = session.split('_')[0]
        session_temp = session.split('_')[1]

        for area_temp in res.area.data:    
            grid_loc = _get_grid_location(monkey_temp, session_temp, area_temp)
            
            # add coordinates to the res
            res['x'].loc[dict(session=session, area=area_temp)] = grid_loc[0]
            res['y'].loc[dict(session=session, area=area_temp)] = grid_loc[1]
        
    # return res
    return res


##% plotting (cortical grid)
def _plot_matrix_best_LR(grid: xr.DataArray,
                monkey: str,
                ax=None, fig=None, title=None, label=None, vmin=None, vmax=None, print_values=False, cmap=plt.cm.get_cmap('Greys')):
    """
    Helper function for plot_on_cortical_grid().

    A plotting method to plot a heatmap grid on the cortical grid of Clement. 
    It loads an image of the sulci of the monkey, and plots the matrix on top of it.

    Parameters
    ----------

    grid : xr.DataArray
        A 19x19 grid with coordinates of the cortical grid of Clement, created by the function init_location_grid.

    monkey : str
        The monkey name, to load the sulci image.

    etc...
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(70, 70))
        show = False

    # ploting helpers
    vmin = np.nanmin(grid) if vmin is None else vmin
    vmax = np.nanmax(grid) if vmax is None else vmax
    # Create blue to yellow colormap
    custom_cmap = plt.cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    # Plot data as scatter
    for x in grid.coords['x']:
        for y in grid.coords['y']:
            value = grid.sel(x=x, y=y).values
            if not np.isnan(value):
                color = custom_cmap(norm(value))
                ax.scatter(x, y, s=20, c=[value], cmap=custom_cmap, vmin=0, vmax=1, alpha=0.8,
                            zorder=5, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=label if label is not None else '', shrink=0.3)
    # bartitle: Best-fitting LR
    cbar.set_label(label if label is not None else 'Best-fitting LR', rotation=270, labelpad=15)

    if print_values:
        # plot values on the grid
        for x in grid.coords['x']:
            for y in grid.coords['y']:
                value = grid.sel(x=x, y=y).values
                if not np.isnan(value):
                    ax.text(x-.2, y+.2, str(int(value)), ha="center", va="center", color="black", fontsize=5, zorder=5, alpha=.5)
                
  
    # plot grey square where its no recording
    #ax.imshow(non_significants, cmap='Greys', alpha=.5, aspect='auto', origin='upper', extent=[-9, 9, -9, 9], vmin=-.1, vmax=.1)

    # plot sulci
    floc = os.path.join(cfg.DATA_PATH, 'recording_sites', f'sulci_{monkey}.png')  # load sulcus png
    img = plt.imread(floc)
    img = np.rot90(img, k=3)  # rotate the image 90 degrees counter-clockwise
    ax.imshow(img, origin='lower', extent=[-10, 10, -10, 10], zorder=10)  # plot sulci, bring to front
    

    # put text on top of the grid (LPFC MCC)
    pos_1 = [2, 8]
    pos_2 = [-4, -8.5]
    ax.text(pos_1[0], pos_1[1], 'MCC', ha="center", va="center", color="black", fontsize=10)
    ax.text(pos_2[0], pos_2[1], 'LPFC', ha="center", va="center", color="black", fontsize=10)

    # text on 'lateral, medial, anterior, posterior'
    ax.set_xlabel('Cortical position (mm)')
    ax.set_ylabel('Cortical position (mm)')
    ax.text(-8, -12, 'post.', ha="center", va="center", color="black")  # labels were swapped (ant./post.)
    ax.text(8, -12, 'ant.', ha="center", va="center", color="black")
    ax.text(-12.5, 8, 'med.', ha="center", va="center", color="black", rotation=90)
    ax.text(-12.5, -8, 'lat.', ha="center", va="center", color="black", rotation=90)

    # show minor ticks at every 1, major at every 2
    ax.set_xticks(np.arange(-8, 9, 1), minor=True)
    ax.set_yticks(np.arange(-8, 9, 1), minor=True)
    ax.set_xticks(np.arange(-8, 9, 2), minor=False)
    ax.set_yticks(np.arange(-8, 9, 2), minor=False)
    #ax.grid(True, which='both', axis='both', linestyle='-', linewidth=1, color='grey', alpha=0.2)

    ax.set(xlim=(-9.5, 9.5), ylim=(-9.5, 9.5), aspect='equal')    
    sns.despine()

    # set title
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_on_cortical_grid_best_LR(
        df: pd.DataFrame,
        column_to_show=None,
        vmin=None, 
        vmax=None, 
        title=None, 
        bar_title=None,
        ax=None,
        print_values=False
        ):
    """
    The full routine to plot the data on the cortical grid of Clement. It will plot the data for each monkey, and each area, on the grid.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data. The dataframe should contain the columns 'monkey', 'session', 'area', and the column to show.

    column_to_show : str
        The column to show on the grid. This is the column that will be plotted on the grid.

    vmin : float
        The minimum value of the colorbar. If None, it will be the minimum value of the data.

    vmax : float
        The maximum value of the colorbar. If None, it will be the maximum value of the data.

    title : str
        The title of the plot.

    bar_title : str
        The title of the colorbar.
    """

    # get the data for the current monkey
    if ax is None:
        cm_to_in = 1/2.54
        fig, ax = plt.subplots(1, 1, figsize=(8*cm_to_in, 12*cm_to_in))
        newfig = True

    # init grid 
    grid = _init_location_grid()
    
    # get the data for the current monkey, set it to the grid value
    locs_already_used = set()
    for (monkey, session, area), sub_df in df.groupby(['monkey', 'session', 'area']):
        # get the location of the current session, area for this monkey
        grid_loc_temp = _get_grid_location(monkey, session, area)
        if np.isnan(grid_loc_temp).any():
            continue

        # add to the grid (if the location is not already used, esle use the better value)
        value_to_show = sub_df[column_to_show].values[0]
        if grid_loc_temp not in locs_already_used:
            grid.loc[dict(x=grid_loc_temp[0], y=grid_loc_temp[1])] = value_to_show
            locs_already_used.add(grid_loc_temp)
        else:  # use the higher value
            grid.loc[dict(x=grid_loc_temp[0], y=grid_loc_temp[1])] = np.max((value_to_show, grid.loc[dict(x=grid_loc_temp[0], y=grid_loc_temp[1])]))  # was np.sum (bug)

    # plot the grid on the subplot
    _plot_matrix_best_LR(grid, monkey, ax=ax, title=f"Monkey {monkey.upper()}", label=bar_title, vmin=vmin, vmax=vmax, print_values=print_values)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if newfig:
        return fig, ax
    else:
        return ax
