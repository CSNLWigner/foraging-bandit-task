"""
Helper functions for the Fig. 3f notebook (population decoding).
"""
import os

import numpy as np
import xarray as xr

from popy.io_tools import load_metadata
from popy.config import RESULTS_PATH


##% data tools
def load_decoding_res_new(folder='basics'):
    xr_res = xr.open_dataset(os.path.join(RESULTS_PATH, 'population_decoding', folder, 'scores.nc'))
    xr_res.load()

    if 'group_target' in xr_res.dims:
        if xr_res.group_target == 'None':
            xr_res = xr_res.mean('group_target')
        else:
            raise ValueError("group_target was used so the plotting method cant handle this.")


    # remove nans along 'session' dimension
    #xr_res = xr_res.dropna('session')
    
    return xr_res


def mask_N_consecutives(mask_xr, N=4):
    """
    Takes a mask (True/False) and sets to False any True value that is not part of N consecutive True values.
    
    Parameters
    ----------
    data : Xarray
        3D Xarray with dimensions (session, time, area)
    N : int
        Number of consecutive non-significant bins to mask
        
    Returns
    -------
    data_masked : Xarray
        3D Xarray with dimensions (session, time, area) with N consecutive non-significant bins masked.
    """

    # find where N consecutive values are False
    mask = np.empty(mask_xr.shape)
    for sess_id in range(mask_xr.shape[0]):
        for area_id in range(mask_xr.shape[2]):
            # get session data
            session_temp = mask_xr[sess_id, :, area_id].values

            consecutive_count = 0
            candidate_ids = []
            unit_signif_ids = []
            for i, if_signif_bin in enumerate(session_temp):
                # if not False, add to count
                if if_signif_bin:
                    consecutive_count += 1
                    candidate_ids.append(i)  # add to candidate indices (may or may not be part of N consecutive bins)
                    if consecutive_count >= N:
                        unit_signif_ids += candidate_ids  # add to list of significant indices
                        candidate_ids = []  # reset candidate ids
                else:  # reset
                    consecutive_count = 0
                    candidate_ids = []
            # create mask for this session (1 if significant, 0 if not) 
            unit_signif = np.zeros_like(session_temp)
            unit_signif[unit_signif_ids] = 1

            mask[sess_id, :, area_id] = unit_signif

    #data_masked = data.where(mask == 1)
    mask = mask.astype(bool)
    mask_xr.data = mask
    return mask_xr


def add_subareas(res):
    metadata = load_metadata()

    subregions = []
    for sess_temp in res.session.values:
        res_temp = res.sel(session=sess_temp)
        # get session metadata
        monkey, session = res_temp.monkey, sess_temp.split('_')[1]
        lpfc_subregion = metadata.loc[(metadata['monkey'] == monkey) & (metadata['session'] == session), 'LPFC_subregion'].values[0]
        subregions.append(lpfc_subregion)

    # add new coordinate along (session, area) dimension
    res = res.assign_coords(lpfc_subregion=('session', subregions))

    return res
