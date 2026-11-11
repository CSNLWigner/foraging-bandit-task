"""
Helper functions for the Fig. 5 notebook.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import popy.config as cfg
from helpers_analysis import fit_mixed_glm, stat_tests_mixed_glm, add_hypothesis


def prepare_2d_scatter_data(data, x_type='within', y_type='alter',
                            session_col='session', feedback_col='feedback',
                            value_col='dV_neural', keep_only_delta=True):
    """Prepare paired session-level scatter data and marginal inputs."""
    if 'dV_signif' not in data.columns:
        data = data.copy()
        data['dV_signif'] = True

    x_data = data[data['type'] == x_type].copy()
    y_data = data[data['type'] == y_type].copy()

    scatter_data = []
    feedback_values = sorted(data[feedback_col].dropna().unique().tolist())

    for session in data[session_col].dropna().unique():
        x_sess_pos = x_data[(x_data[session_col] == session) & (x_data[feedback_col] == 1)]
        y_sess_pos = y_data[(y_data[session_col] == session) & (y_data[feedback_col] == 1)]
        value_sess = data[data[session_col] == session]

        if x_sess_pos.empty or y_sess_pos.empty or value_sess.empty:
            continue

        signif_same = bool(x_sess_pos['dV_signif'].iloc[0])
        signif_alter = bool(y_sess_pos['dV_signif'].iloc[0])
        value_coding = value_sess['value_coding'].iloc[0] if 'value_coding' in value_sess.columns else None

        for feedback in feedback_values:
            x_val = x_data[(x_data[session_col] == session) & (x_data[feedback_col] == feedback)][value_col].values
            y_val = y_data[(y_data[session_col] == session) & (y_data[feedback_col] == feedback)][value_col].values

            if len(x_val) == 0 or len(y_val) == 0:
                continue

            scatter_data.append({
                'x': x_val[0],
                'y': y_val[0],
                'feedback': str(int(feedback)),
                'session': session,
                'signif_same': signif_same,
                'signif_alter': signif_alter,
                'value_coding': value_coding,
            })

    scatter_df = pd.DataFrame(scatter_data)

    if keep_only_delta and not scatter_df.empty and 'value_coding' in scatter_df.columns:
        is_delta = scatter_df['value_coding'].isin(['Delta rule (loose)', 'Delta rule (strict)'])
        scatter_df = scatter_df[is_delta].copy()

    return {
        'scatter_df': scatter_df,
        'x_data': x_data,
        'y_data': y_data,
        'n_sessions': len(np.unique(x_data[session_col])) if len(x_data) > 0 else 0,
    }



def get_session_mean_value(res, session_summary):
    # get mean of trials (i.e. mean of all trials in a session) for each monkey, session, subregion, feedback, type
    results_df_mean = []
    for (monkey, session, subregion, type_), group in res.groupby(['monkey', 'session', 'subregion', 'type']):
        '''hypothesis_of_session = session_summary[(session_summary['monkey'] == monkey) &
                                            (session_summary['session'] == session) &
                                            (session_summary['subregion'] == subregion)]['hypothesis'].values[0]
        
        if not hypothesis_of_session == 'Delta rule (strict)':
            # skip these so that we only look at value related activity
            continue'''

        # compute significance of modulation between positive and negative feedback for this session, subregion, type
        dVs_pos = group[(group['feedback'] == 1) & (group['type'] == type_)]['dV_neural'].values
        dVs_neg = group[(group['feedback'] == 0) & (group['type'] == type_)]['dV_neural'].values
        U, p_value = stats.mannwhitneyu(dVs_pos, dVs_neg, alternative='two-sided')
        if_signif = p_value < 0.05

        # value coding hypothesis of this monkey, session and subregion:
        value_coding_hypothesis = session_summary[(session_summary['monkey'] == monkey) &
                                            (session_summary['session'] == session) &
                                            (session_summary['subregion'] == subregion)]['hypothesis'].values[0]

        # get mean of dV_neural per feedback type
        for feedback in group['feedback'].unique():
            dV_neural = group[group['feedback'] == feedback]['dV_neural'].mean()
            V_neural = group[group['feedback'] == feedback]['V_neural'].mean()
            V_neural_p1 = group[group['feedback'] == feedback]['V_neural_p1'].mean()
            
            # , drop trial_id
            row_to_add = {}
            row_to_add['monkey'] = monkey
            row_to_add['session'] = session
            row_to_add['subregion'] = subregion
            row_to_add['type'] = type_
            row_to_add['feedback'] = feedback
            row_to_add['dV_neural'] = dV_neural
            row_to_add['V_neural'] = V_neural
            row_to_add['V_neural_p1'] = V_neural_p1
            row_to_add['dV_signif'] = if_signif
            row_to_add['value_coding'] = value_coding_hypothesis

            results_df_mean.append(row_to_add)

    # to df
    results_df_mean = pd.DataFrame(results_df_mean)

    return results_df_mean

def get_data(floc=None):
    if floc is None:
        path = os.path.join(cfg.RESULTS_PATH, 'population_decoding', 'across_target_before_fb', 'behav.pkl')
    else:        
        path = os.path.join(floc, 'behav.pkl')
        
    results_df = pd.read_pickle(path)
    # sort by monkey session trial_id target type
    results_df = results_df.sort_values(by=['monkey', 'session', 'subregion', 'trial_id'])
    results_df = results_df.dropna()
    return results_df

def merge_alter_values(results_df):
    """Merge 'alter' type rows by averaging their values for each trial, so that we only have one."""
    # for each (monkey, session, subregion, trial_id), replace the two 'type'=='alter' rows with one that has these attributes, and averages everything else

    results_df_within = results_df[results_df['type'] == 'within']
    results_df_alter = results_df[results_df['type'] == 'alter']

    results_df_alter_means = []
    for (monkey, session, subregion, trial_id), group in results_df_alter.groupby(['monkey', 'session', 'subregion', 'trial_id']):
        mean_row = {}
        mean_row['monkey'] = monkey
        mean_row['session'] = session
        mean_row['subregion'] = subregion
        mean_row['trial_id'] = trial_id
        mean_row['type'] = 'alter'
        mean_row['feedback'] = group['feedback'].values[0]  # or .mean() if you want the average
        mean_row['dV_neural'] = group['dV_neural'].mean()
        mean_row['V_neural'] = group['V_neural'].mean()
        mean_row['V_neural_p1'] = group['V_neural_p1'].mean()

        results_df_alter_means.append(mean_row)
    results_df_alter_means = pd.DataFrame(results_df_alter_means)

    # Combine the two dataframes
    results_df_combined = pd.concat([results_df_within[results_df_alter_means.columns.tolist()], results_df_alter_means], ignore_index=True)

    # feedback is int

    results_df = results_df_combined
    results_df = results_df.sort_values(by=['monkey', 'session', 'subregion', 'trial_id'])

    # reset index
    results_df = results_df.reset_index(drop=True)

    return results_df

def extract_hypothesis(results_df_mean, detailed=True):
    """
    """

    stats_df = []
    for (monkey, subregion), results_df_mean_temp in results_df_mean.groupby(['monkey', 'subregion']):
        hypothesis_count = {
            'Foraging': 0,
            'Standard RL': 0,
            'Inferential RL': 0,
            'none': 0,
        }

        for _, group in results_df_mean_temp.groupby('session'):
            # get p-values for within and alter targets to tell if the difference between positive and negative feedback is significant
            '''dVs_pos_within = group[(group['feedback'] == 1) & (group['type'] == 'within')]['dV_neural'].values
            dVs_neg_within = group[(group['feedback'] == 0) & (group['type'] == 'within')]['dV_neural'].values
            U_same, p_value_same = stats.mannwhitneyu(dVs_pos_within, dVs_neg_within, alternative='two-sided')

            dVs_pos_alter = group[(group['feedback'] == 1) & (group['type'] == 'alter')]['dV_neural'].values
            dVs_neg_alter = group[(group['feedback'] == 0) & (group['type'] == 'alter')]['dV_neural'].values
            U_alter, p_value_alter = stats.mannwhitneyu(dVs_pos_alter, dVs_neg_alter, alternative='two-sided')'''

            # get p-values for within and alter targets to tell if the V_t is significanlys correlated with V_t+1
            '''Vs = group[group['type'] == 'within']['V_neural'].values
            Vp1s = group[group['type'] == 'within']['V_neural_p1'].values
            _, p_value_corr_within_pos = stats.spearmanr(Vs, Vp1s)'''
            # intercept greater for positive?
            # slope between 0 and 1 signifcantly for both pos and neg?

        
            # Determine hypothesis based on p-values and U statistics
            dV_signif_within = group[group['type'] == 'within']['dV_signif'].values[0]
            dV_signif_alter = group[group['type'] == 'alter']['dV_signif'].values[0]
            pos_coding_within = group[(group['type'] == 'within') & (group['feedback'] == 1)]['dV_neural'].values[0] > group[(group['type'] == 'within') & (group['feedback'] == 0)]['dV_neural'].values[0]
            pos_coding_alter = group[(group['type'] == 'alter') & (group['feedback'] == 1)]['dV_neural'].values[0] > group[(group['type'] == 'alter') & (group['feedback'] == 0)]['dV_neural'].values[0]
            #is_value = p_value_corr_within_pos < 0.05
    
            if dV_signif_within:
                if not pos_coding_within:  # Problem handling 1: significant within but in the wrong direction
                    hypothesis = 'reverse within'
                elif dV_signif_alter:
                    if pos_coding_within and pos_coding_alter:  # H1 Foraging: significant in both, and positive coding in both
                        hypothesis = 'Foraging'
                    elif pos_coding_within and not pos_coding_alter:  # H2 Inferential RL: significant in both, but negative coding in alter target
                        hypothesis = 'Inferential RL'
                    else:
                        raise ValueError("Unexpected combination of significance and coding direction for within and alter targets.")
                elif not dV_signif_alter:
                    if pos_coding_within:  # H3 Standard RL: significant only in within target
                        hypothesis = 'Standard RL'
                    else:
                        hypothesis = 'reverse within'
            else:
                if dV_signif_alter:
                    hypothesis = 'only signif alter'
                else:
                    hypothesis = 'none'

            # merge all in the none that are not in the main hypotheses
            if not detailed:
                if hypothesis not in ['Foraging', 'Standard RL', 'Inferential RL']:
                    hypothesis = 'none'        
            
            # init new hypothesis in count dict if not there yet
            if hypothesis not in hypothesis_count.keys():
                hypothesis_count[hypothesis] = 0

            # add to count for this hypothesis
            hypothesis_count[hypothesis] += 1

        # Create a summary row for this monkey and subregion

        for hyp, count in hypothesis_count.items():
            stats_df.append({
                'monkey': monkey,
                'subregion': subregion,
                'hypothesis': hyp,
                'count': count
            })

    stats_df = pd.DataFrame(stats_df)

    return stats_df

def make_session_summary(res):
    # extract hypothesis per session
    # add hypothesis columns (empty for now, full of nans)
    session_summary = []

    for (monkey, session, subregion), subdf in res.groupby(['monkey', 'session', 'subregion']):
        # get hypothesis
        data_temp = pd.DataFrame({
            'A': subdf[subdf['type'] == 'within']['V_neural_p1'].values,
            'B': subdf[subdf['type'] == 'within']['V_neural'].values,
            'C': subdf[subdf['type'] == 'within']['feedback'].values,
        })

        results_mixed = fit_mixed_glm(data_temp)
        test_results = stat_tests_mixed_glm(results_mixed)
        hypothesis = add_hypothesis(test_results)

        # write hypothesis back to res
        session_summary.append({
            'monkey': monkey,
            'session': session,
            'subregion': subregion,
            'hypothesis': hypothesis,
        })

    session_summary = pd.DataFrame(session_summary)
    return session_summary
