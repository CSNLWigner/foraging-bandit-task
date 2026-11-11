"""
Helper functions for the Fig. 3c-e notebook (single-unit GLM).
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

from popy.decoding.decoder_tools import mask_N_consecutives
import popy.config as cfg


##% helper functions
def load_res(fname, target=None, add_target_to_fname=True):
    floc = os.path.join(
        cfg.RESULTS_PATH,
        "glm",
        f"{fname}_target_{target}" if add_target_to_fname else fname,
        "scores.nc",
    )
    print(f"loading {floc}")

    ds = xr.open_dataset(floc, engine="netcdf4")
    ds.close()
    return ds.copy("deep")


def add_lpfc_subregions(ds_fb_value_full):
    
    from popy.io_tools import load_metadata

    # adding subregions: MCC, dLPFC, vLPFC
    metadata = load_metadata()
    subregions = []
    for unit in ds_fb_value_full.unit.values:
        unit_data = ds_fb_value_full.sel(unit=unit)
        monkey = unit_data.monkey.values
        session = unit_data.session.values
        area = unit_data.area.values

        if area == "LPFC":
            lprc_subregion = metadata.loc[
                (metadata["monkey"] == monkey) & 
                (metadata["session"] == session), "LPFC_subregion"].values[0]
            
            subregions.append(lprc_subregion)

        elif area == "MCC":
            subregions.append('MCC')
        else:
            raise ValueError(f"Unknown area: {area}")

    # add subregions to the dataset
    ds_fb_value_full = ds_fb_value_full.assign_coords(
        subregion=("unit", subregions)
    )
    return ds_fb_value_full


def extract_significant_units(ds_fb_value_full, plim=0.05, n_consecutives=2, t_feedback=[0, 1], t_history=[-1, 0]):

    # get significant times and units
    signifs_fb_all = ds_fb_value_full.sel(time=slice(*t_feedback)).CPDs_p_vals_feedback < plim
    signifs_val_all = ds_fb_value_full.sel(time=slice(*t_history)).CPDs_p_vals_stay_value < plim

    signifs_fb = mask_N_consecutives(signifs_fb_all, N=n_consecutives, non_signif_coding='0')  # mask 3 consecutives
    signifs_val = mask_N_consecutives(signifs_val_all, N=n_consecutives, non_signif_coding='0')  # mask 3 consecutives

    # is there ANY time point for a given unit that is significant?
    if_signifs_fb = signifs_fb.sum("time") > 0
    if_signifs_val = signifs_val.sum("time") > 0

    signif_groups = []
    for if_signif_fb, if_signif_value in zip(if_signifs_fb.values, if_signifs_val.values):
        if (not if_signif_fb) and (not if_signif_value):
            signif_groups.append('None')
        elif if_signif_fb and if_signif_value:
            signif_groups.append('Both')
        elif if_signif_fb:
            signif_groups.append('Fb only')
        elif if_signif_value:
            signif_groups.append('Value only')

    # get corresponding tvals
    tvals_fb_all = ds_fb_value_full.sel(time=slice(*t_feedback)).t_vals_feedback
    #tvals_fb_all = tvals_fb_all.where(signifs_fb, np.nan)  # mask non-significant values
    tvals_fb_max = tvals_fb_all.max("time")
    tvals_fb_min = tvals_fb_all.min("time")

    tvals_fb_best = []
    for tval_fb_max, tval_fb_min in zip(tvals_fb_max.values, tvals_fb_min.values):
        if tval_fb_max >= np.abs(tval_fb_min):
            tvals_fb_best.append(tval_fb_max)
        else:
            tvals_fb_best.append(tval_fb_min)

    tvals_val_all = ds_fb_value_full.sel(time=slice(*t_history)).t_vals_stay_value
    #tvals_val_all = tvals_val_all.where(signifs_val, np.nan)
    tvals_val_max = tvals_val_all.max("time")
    tvals_val_min = tvals_val_all.min("time")

    tvals_val_best = []
    for tval_val_max, tval_val_min in zip(tvals_val_max.values, tvals_val_min.values):
        if tval_val_max >= np.abs(tval_val_min):
            tvals_val_best.append(tval_val_max)
        else:
            tvals_val_best.append(tval_val_min)

    df_unit_significances = pd.DataFrame(
        {
            "unit": ds_fb_value_full.unit.values,
            "monkey": ds_fb_value_full.monkey,
            "session": ds_fb_value_full.session,
            "area": ds_fb_value_full.area,
            "subregion": ds_fb_value_full.subregion,
            #"channel": ds_fb_value_full.channel,
            "signif_group": signif_groups,
            "t_vals_feedback": tvals_fb_best,
            "t_vals_value": tvals_val_best,
        }
    )

    # remove 1% outliers from t_vals_feedback and t_vals_stay_value
    '''outliers_fb = df_unit_significances.t_vals_feedback.abs() > df_unit_significances.t_vals_feedback.abs().quantile(0.99)
    outliers_value = df_unit_significances.t_vals_value.abs() > df_unit_significances.t_vals_value.abs().quantile(0.99)

    df_unit_significances = df_unit_significances[~(outliers_fb | outliers_value)]
    print(f"Number of units removed due to outliers: {(outliers_fb | outliers_value).sum()}")'''

    # return
    return df_unit_significances


def extract_number_of_signifs(df_unit_significances):
    prop_units = []
    for (monkey, session, area), sub_df in df_unit_significances.groupby(["monkey", "session", "area"]):
            # get prop signif units for the current session (for given regressors, for given times(?) )
            n_units = len(sub_df)
            n_signif_fb = len(sub_df.loc[(sub_df['signif_group'] == 'Fb only') | (sub_df['signif_group'] == 'Both')])
            n_signif_value = len(sub_df.loc[(sub_df['signif_group'] == 'Value only') | (sub_df['signif_group'] == 'Both')])

            '''tval_fb_max = sub_df.loc[(sub_df['signif_group'] == 'Fb only') | (sub_df['signif_group'] == 'Both')]['t_vals_feedback'].max()
            tval_fb_min = sub_df.loc[(sub_df['signif_group'] == 'Fb only') | (sub_df['signif_group'] == 'Both')]['t_vals_feedback'].min()
            tval_fb_best = max_abs(tval_fb_max, tval_fb_min)

            tval_value_max = sub_df.loc[(sub_df['signif_group'] == 'Value_only only') | (sub_df['signif_group'] == 'Both')]['t_vals_value'].max()
            tval_value_min = sub_df.loc[(sub_df['signif_group'] == 'Value_only only') | (sub_df['signif_group'] == 'Both')]['t_vals_value'].min()
            tval_value_best = max_abs(tval_value_max, tval_value_min)'''

            # save to dict
            prop_units.append(
                {
                    "monkey": monkey,
                    "session": str(session),
                    "area": area,
                    'subregion': sub_df.subregion.values[0],
                    "n_units": n_units,
                    "n_signif_feedback": n_signif_fb,
                    "n_signif_value": n_signif_value,
                }
            )

    return pd.DataFrame(prop_units)
