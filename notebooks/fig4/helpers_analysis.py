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


def simple_glm_weights(behav_new):
    # Data setup
    weights_all = []
    for monkey in behav_new.monkey.unique():
        for subregion in behav_new.subregion.unique():
            behav_temp = behav_new.loc[(behav_new.monkey == monkey) & (behav_new.subregion == subregion)]

            for session in behav_temp.session.unique():
                behav = behav_temp.loc[behav_temp.session == session].copy()

                y = behav.V_t.values
                X = np.array([behav[f'R_{i}'].values for i in range(1, 9)]).T
                # add one more column for the randomly shuffled values

                # Add intercept for statsmodels
                X_sm = sm.add_constant(X)
                model = sm.OLS(y, X_sm)
                results = model.fit()

                # Extract weights and p-values (skip intercept)
                weights = results.params[1:]
                pvals = results.pvalues[1:]
                
                res_temp = {f'R_{i}': results.params[i] for i in range(1, len(weights) + 1)}
                res_temp['monkey'] = monkey
                res_temp['session'] = session
                res_temp['subregion'] = subregion
                weights_all.append(res_temp)

    weights_all = pd.DataFrame(weights_all)
    return weights_all

def single_permutation_cpd(y, X_full, i, seed=None):
    """Calculate CPD for a single permutation"""
    if seed is not None:
        np.random.seed(seed)
    
    X_perm = X_full.copy()
    X_perm[:, i] = np.random.permutation(X_perm[:, i])
    
    # Full model with permuted data
    X_perm_sm = sm.add_constant(X_perm)
    model_perm_full = sm.OLS(y, X_perm_sm).fit()
    r2_perm_full = model_perm_full.rsquared
    
    # Reduced model with permuted data
    X_perm_reduced = np.delete(X_perm, i, axis=1)
    X_perm_reduced_sm = sm.add_constant(X_perm_reduced)
    model_perm_reduced = sm.OLS(y, X_perm_reduced_sm).fit()
    r2_perm_reduced = model_perm_reduced.rsquared
    
    # Calculate CPD for this permutation
    cpd_perm = ((r2_perm_full - r2_perm_reduced) / r2_perm_full) * 100
    return cpd_perm

def process_single_session(behav, n_perms=100):
    y = behav.V_t.values
    X_full = np.array([behav[f'R_{i}'].values for i in range(1, 9)]).T
    
    # Calculate full model R²
    X_full_sm = sm.add_constant(X_full)
    model_full = sm.OLS(y, X_full_sm).fit()
    r2_full = model_full.rsquared
    
    cpd_temp = {}
    #pvals_temp = {}
    
    # Process each regressor
    for i in range(X_full.shape[1]):                
        # Calculate actual CPD
        X_reduced = np.delete(X_full, i, axis=1)
        X_reduced_sm = sm.add_constant(X_reduced)
        model_reduced = sm.OLS(y, X_reduced_sm).fit()
        r2_reduced = model_reduced.rsquared
        cpd_actual = ((r2_full - r2_reduced) / r2_full) * 100
        
        # Parallel permutation testing with different seeds for reproducibility
        seeds = np.random.randint(0, 2**31, n_perms)
        cpd_perm = []
        for n_i in range(n_perms):
            cpd_perm.append(single_permutation_cpd(y, X_full, i, seed=seeds[n_i]))

        cpd_perm = np.array(cpd_perm)
        pval = (np.sum(cpd_perm >= cpd_actual) + 1) / (n_perms + 1)
        
        key = f'R_{i+1}'
        cpd_temp[f'cpd_{key}'] = cpd_actual
        cpd_temp[f'pval_{key}'] = pval
        
    #cpd_temp.update(pvals_temp)
    cpd_temp['monkey'] = behav['monkey'].unique()[0]  # Assuming single monkey per session
    cpd_temp['session'] = behav['session'].unique()[0]  # Assuming single session per behavior
    cpd_temp['subregion'] = behav['subregion'].unique()[0]  # Assuming single subregion per behavior

    return cpd_temp

def calculate_cpds(y, X):
    """Calculate CPDs (Change in Explained Variance) for a given dependent variable y and independent variables X."""
    import statsmodels.api as sm

    # add random column to X_full
    random_col = np.random.permutation(X[:, -1])
    X_full_random = np.column_stack((X, random_col))
    # Add intercept for statsmodels
    X_full_random = sm.add_constant(X_full_random)  # Add intercept for statsmodels

    model_full = sm.OLS(y, X_full_random).fit()
    r2_full = model_full.rsquared

    # Leave out random column for reduced model, i.e. use only the original regressors
    X_reduced = X
    X_reduced = sm.add_constant(X_reduced)  # Add intercept for statsmodels
    model_reduced = sm.OLS(y, X_reduced).fit()
    r2_reduced = model_reduced.rsquared
    # Calculate change in explained variance, in percentage points
    cpd = ((r2_full - r2_reduced) / r2_full) * 100

    return cpd

def create_permutation_null(behav_temp, n_perms=1000):
    """Create null by permuting the dependent variable"""
    

    perm_cpds = []
    for perm in range(n_perms):
        
        session_cpds = []
        for session in behav_temp.session.unique():
            behav = behav_temp.loc[behav_temp.session == session].copy()
            
            # PERMUTE THE DEPENDENT VARIABLE
            y = behav.V_t.values  # This breaks real relationships
            X = np.array([behav[f'R_{i}'].values for i in range(1, 9)]).T
            
            # Calculate CPDs with permuted y
            cpd = calculate_cpds(y, X)  # your existing CPD function
            session_cpds.append(cpd)
        
        # Average across sessions for this group
        group_mean_cpds = np.mean(session_cpds, axis=0)
        perm_cpds.append(group_mean_cpds)

    return np.array(perm_cpds)

def add_corrected_pvals(cpd_all):
    from statsmodels.stats.multitest import multipletests

    # Get p-value columns
    pval_cols = [col for col in cpd_all.columns if col.startswith('pval_') and not col.endswith('corrected')]

    # Create new columns for corrected p-values
    for col in pval_cols:
        cpd_all[f'{col}_corrected'] = np.nan

    # Apply correction within each group (monkey × subregion)
    for monkey in cpd_all.monkey.unique():
        for subregion in cpd_all.subregion.unique():
            # Get mask for this group
            group_mask = (cpd_all.monkey == monkey) & (cpd_all.subregion == subregion)
            
            # Extract p-values for this group
            group_pvals = cpd_all.loc[group_mask, pval_cols].values
            
            # Apply correction within this group
            rejected, pvals_corrected, _, _ = multipletests(
                group_pvals.flatten(), method='fdr_bh'
            )
            
            # Reshape back to original shape
            pvals_corrected = pvals_corrected.reshape(group_pvals.shape)
            
            # Write back corrected p-values
            for i, col in enumerate(pval_cols):
                cpd_all.loc[group_mask, f'{col}_corrected'] = pvals_corrected[:, i]

    ''' # Now use the corrected p-values
    pval_cols_corrected = [f'{col}_corrected' for col in pval_cols]'''

    return cpd_all

def get_cpd_all(behav_new, n_perms):

    # Get all unique monkey-subregion combinations
    # Get all unique monkey-subregion-session combinations
    session_behavs = []
    for monkey in behav_new.monkey.unique():
        for subregion in behav_new.subregion.unique():
            behav_temp = behav_new.loc[(behav_new.monkey == monkey) & (behav_new.subregion == subregion)]
            for session in behav_temp.session.unique():
                behav_temp = behav_new.loc[(behav_new.monkey == monkey) & (behav_new.subregion == subregion)]
                behav = behav_temp.loc[behav_temp.session == session].copy()

                session_behavs.append(behav)

    # Process all combinations in parallel

    print(f"Using all available CPU cores for parallel processing")
    cpd_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_single_session)(behav, n_perms)
        for behav in session_behavs
    )
    '''
    cpd_results = []
    for behav in session_behavs:
        cpd = process_single_session(behav, n_perms)
        cpd_results.append(cpd)
    '''

    cpd_all = pd.DataFrame(cpd_results)
    #cols_new = ['monkey', 'subregion', 'session'] + [col for col in cpd_all.columns if col not in ['monkey', 'subregion', 'session']]
    #cpd_all = cpd_all[cols_new]

    print("Analysis complete!")

    # correct pvals per group with FDR
    cpd_all = add_corrected_pvals(cpd_all)

    # save results
    #cpd_all.to_pickle('results/cpd_all.pkl')

    return cpd_all

##% Plotting Functions

def print_stat_header(fig_label, data):
    """Print a header line for the statistics of a figure panel: figure, monkey(s), subregion(s)."""
    monkeys = ', '.join(sorted(str(m).upper() for m in data['monkey'].unique())) if 'monkey' in data.columns else 'n/a'
    subregions = ', '.join(sorted(str(s) for s in data['subregion'].unique())) if 'subregion' in data.columns else 'n/a'
    print(f"\n===== {fig_label} | monkey: {monkeys} | subregion: {subregions} =====")


def wilcoxon_signed_rank(x):
    """Two-sided Wilcoxon signed-rank test of x against 0.

    Returns the test statistic W, the p-value, the matched-pairs rank-biserial correlation (effect size)
    and the number of non-zero values used by the test.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    res = stats.wilcoxon(x, alternative='two-sided')

    nonzero = x[x != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    r_rb = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()

    return res.statistic, res.pvalue, r_rb, len(nonzero)


def bootstrap_median_ci(x, n_resamples=10000, confidence_level=0.95, seed=0):
    """Percentile bootstrap confidence interval of the median (resampling sessions)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    res = stats.bootstrap((x,), np.median, n_resamples=n_resamples, confidence_level=confidence_level,
                          method='percentile', random_state=np.random.default_rng(seed))
    return res.confidence_interval.low, res.confidence_interval.high


def create_r_style_plot(ax=None, data=None, x_col=None, y_col=None, title=None, session_col='session', paper_format=False, fig_label='Fig. 4e', n_comparisons=3):
    """Create R-style plot with violin, box, and jittered points

    Args:
        ax: matplotlib axis
        data: DataFrame with the data
        x_col: column name for x-axis categories
        y_col: column name for y-axis values
        title: plot title
        session_col: column name for session identifier (optional, for connecting points)
        fig_label: figure panel name used in the printed statistics
        n_comparisons: number of tests for the Bonferroni correction (default: 3 regions)
    """
    palette = COLORS
    axis = ax
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 1.5
        w = 2
        s = 3

    else:
        plt.rcParams.update({'font.size': 18})
        h = 3  # in cm
        w = 4  # in cm
        s = 10

    
    # distribution of pos and neg feedback along neural value
    if axis is None:
        fig, ax = plt.subplots(1, 1, figsize=(w, h))


    # Get unique categories
    # sort by x_col
    data = data.copy().sort_values(by=x_col)
    categories = data[x_col].unique()
    x_positions = np.arange(len(categories))
    
    x_offset = .2
    s = 10
    lw = .4 
    
    # Check if we'll be drawing connecting lines
    will_connect_lines = False
    if session_col is not None and len(categories) == 2:
        sessions_cat1 = set(data.loc[data[x_col] == categories[0], session_col])
        sessions_cat2 = set(data.loc[data[x_col] == categories[1], session_col])
        common_sessions = sessions_cat1.intersection(sessions_cat2)
        
        valid_sessions = []
        for session in common_sessions:
            count_cat1 = len(data.loc[(data[x_col] == categories[0]) & (data[session_col] == session)])
            count_cat2 = len(data.loc[(data[x_col] == categories[1]) & (data[session_col] == session)])
            if count_cat1 == 1 and count_cat2 == 1:
                valid_sessions.append(session)
        
        will_connect_lines = len(valid_sessions) > 0
    
    # Create violin plots (distributions)
    violin_positions = []
    for i in range(len(categories)):
        if will_connect_lines and i == 0:
            # First group: distribution on left
            violin_positions.append(x_positions[i] - x_offset)
        else:
            # Default: distribution on right
            violin_positions.append(x_positions[i] + x_offset)
    
    # Create half violin plots (distributions) - flat side towards boxplot
    for i, cat in enumerate(categories):
        y_data = data.loc[data[x_col] == cat, y_col].values
        
        # Create violin plot data
        density = stats.gaussian_kde(y_data)
        
        # Create y range for the distribution
        y_min, y_max = y_data.min(), y_data.max()
        y_range = np.linspace(y_min - 0.1 * (y_max - y_min), 
                             y_max + 0.1 * (y_max - y_min), 200)
        density_values = density(y_range)
        
        # Normalize density values to desired width
        density_values = density_values / density_values.max() * 0.15
        
        # Determine which side the distribution should be on
        if will_connect_lines and i == 0:
            # First group: distribution on left (away from points/lines)
            x_dist = violin_positions[i] - density_values
            x_null = violin_positions[i]  + .05
        else:
            # Default: distribution on right (away from points)
            x_dist = violin_positions[i] + density_values
            x_null = violin_positions[i] - .05
        
        # Fill the half violin
        ax.fill_betweenx(y_range, x_null, x_dist, 
                        color=palette[cat], alpha=0.3, 
                        edgecolor='black', linewidth=2*lw)
    
    # Create boxplots - centered
    box_data = [data.loc[data[x_col] == cat, y_col].values for cat in categories]
    bp = ax.boxplot(box_data, positions=x_positions, widths=0.2, patch_artist=True,
                   boxprops=dict(alpha=0.7),
                   showfliers=False,
                   medianprops=dict(color='black'),
                   #whiskerprops=dict(linewidth=1.5),
                   #capprops=dict(linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='black', markersize=3, alpha=0.5))
    
    # Color the boxes
    colors = [palette[cat] for cat in categories]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Store jitter positions for connecting lines
    jitter_positions = {}
    
    # Add jittered points
    for i, cat in enumerate(categories):
        y_data = data.loc[data[x_col] == cat, y_col].values
        if 'signif' in data.columns:
            signifs = data.loc[data[x_col] == cat, 'signif'].values
            colors = [palette[cat] if signif else 'white' for signif in signifs]
        else:
            colors = [palette[cat]] * len(y_data)
        
        # Determine jitter position based on whether we're connecting lines
        if will_connect_lines and i == 0:
            # First group: points on right when connecting lines
            x_jitter = np.random.normal(x_positions[i] + x_offset, 0.02, size=len(y_data))
        else:
            # Default: points on left
            x_jitter = np.random.normal(x_positions[i] - x_offset, 0.02, size=len(y_data))
        
        ax.scatter(x_jitter, y_data, alpha=0.6, s=s, c=colors,
                   edgecolor='black', linewidth=lw)
        
        # Store jitter positions if we need to connect points
        if session_col is not None:
            session_data = data.loc[data[x_col] == cat, session_col].values
            jitter_positions[cat] = dict(zip(session_data, zip(x_jitter, y_data)))
    
    # Connect points between categories if session_col is provided and we have exactly 2 categories
    if will_connect_lines:
        # Draw connecting lines for valid sessions
        for session in valid_sessions:
            if session in jitter_positions[categories[0]] and session in jitter_positions[categories[1]]:
                x1, y1 = jitter_positions[categories[0]][session]
                x2, y2 = jitter_positions[categories[1]][session]
                ax.plot([x1, x2], [y1, y2], '-', color='gray', alpha=0.2, zorder=0, lw=lw)
    
    # Perform t-test and add significance
    if len(categories) == 2:
        group1_data = data.loc[data[x_col] == categories[0], y_col]
        group2_data = data.loc[data[x_col] == categories[1], y_col]
        print_stat_header(fig_label, data)

        if session_col is not None and session_col in data.columns:
            # paired test: each session contributes one value per category
            pair_keys = [col for col in ['monkey', session_col] if col in data.columns]
            paired = data.pivot_table(index=pair_keys, columns=x_col, values=y_col).dropna()
            diffs = paired[categories[1]] - paired[categories[0]]
            stat, p_value, r_rb, n_used = wilcoxon_signed_rank(diffs.values)
            stat_name = 'W'

            print(f"  Group test: two-sided Wilcoxon signed-rank test, paired by session, {y_col} of {categories[1]} - {categories[0]} ({x_col})")
            print(f"  n = {len(diffs)} sessions ({n_used} non-zero differences); median difference = {np.median(diffs):.3f} "
                  f"(IQR {np.percentile(diffs, 25):.3f} to {np.percentile(diffs, 75):.3f})")
            ci_low, ci_high = bootstrap_median_ci(diffs.values)
            print(f"  95% CI of the median difference (bootstrap over sessions, 10000 resamples): [{ci_low:.3f}, {ci_high:.3f}]")
            print(f"  W = {stat:.1f}, p = {p_value:.3e}, p (Bonferroni x{n_comparisons}) = {min(p_value * n_comparisons, 1):.3e}")
            print(f"  Effect size: matched-pairs rank-biserial r = {r_rb:.3f}")
        else:
            # no session identifier: groups can not be paired, fall back to an unpaired test
            stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            stat_name = 'U'
            n1, n2 = len(group1_data), len(group2_data)

            print(f"  Group test: two-sided Mann-Whitney U test (unpaired), {y_col} of {categories[0]} vs. {categories[1]} ({x_col})")
            print(f"  n = {n1} vs. {n2}")
            print(f"  U = {stat:.1f}, p = {p_value:.3e}, p (Bonferroni x{n_comparisons}) = {min(p_value * n_comparisons, 1):.3e}")
            print(f"  Effect size: rank-biserial r = {1 - 2 * stat / (n1 * n2):.3f}")

        if 'signif' in data.columns and session_col is not None:
            signif_sessions = data.groupby([col for col in ['monkey', session_col] if col in data.columns])['signif'].first()
            print(f"  Per-session test (dot fill): two-sided Mann-Whitney U test on single trials, uncorrected: "
                  f"{signif_sessions.sum()}/{len(signif_sessions)} sessions with p < 0.05")

        # Position for significance bar
        y_max = data[y_col].max()
        y_pos = y_max + 0.1 * abs(y_max)
        
        # Add significance annotation
        if p_value < 0.001:
            significance = '***, {}={:.2f}, p={}'.format(stat_name, stat, p_value)
        elif p_value < 0.01:
            significance = '**, {}={:.2f}, p={}'.format(stat_name, stat, p_value)
        elif p_value < 0.05:
            significance = '*, {}={:.2f}, p={}'.format(stat_name, stat, p_value)
        else:
            significance = 'n.s., {}={:.2f}, p={}'.format(stat_name, stat, p_value)
        
        # Add the bar and text for p-value
        ax.plot([0, 1], [y_pos, y_pos], '-k', lw=2*lw)
        ax.text(0.5, y_pos + 0.02 * abs(y_max), f'{significance} (n={len(group1_data)})', 
                ha='center', va='bottom', fontsize=8)
    
    # Formatting
    ax.axhline(0, linestyle='--', alpha=0.7, color='black', lw=2*lw, zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Unrewarded', 'Rewarded'], rotation=20)
    ax.set_ylabel('Change in neural value \n$\Delta V = V_t - V_{t-1}$', fontsize=8)
    ax.set_xlabel('')
    ax.set_title(title, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if axis is None:
        return fig, ax
    else:
        return ax


def plot_Vt_per_sequence(behav_new, paper_format=False, ylim=None, show_datapoints=True, ax=None, showfliers=True, fig_label='Fig. 4c', n_comparisons=3):
    axis_original = ax
    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 1.5 
        w = 2
        s = 3
    else:
        plt.rcParams.update({'font.size': 18})
        h = 3  # in cm
        w = 4  # in cm
        s = 10

    
    # distribution of pos and neg feedback along neural value
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(w, h))

    # Example: get 10 colors from your scale
    colors = value_gradient(8)

    # remove 1% outliers in V_t
    '''V_t_threshold = np.percentile(np.abs(behav_new.V_t.values), 99)
    behav_new = behav_new[np.abs(behav_new.V_t.values) < V_t_threshold]'''

    df = pd.DataFrame({
        #'trial_id': behav_new.trial_id.values,
        'V_t': behav_new.V_t.values,
        'fb_sequence': behav_new.fb_sequence.values,
        'fb_sequence_last_2': behav_new.fb_sequence.values % 4,
    })

    label_mapping_long = {0: '0\n0\n0', 1: '0\n0\n1', 2:'0\n1\n0', 3:'0\n1\n1', 4: '1\n0\n0', 5: '1\n0\n1', 6:'1\n1\n0', 7:'1\n1\n1'}
    label_mapping_short = {0: '0\n0', 1: '0\n1', 2:'1\n0', 3:'1\n1'}
    df['fb_sequence_str'] = df['fb_sequence'].map(label_mapping_long)
    df['fb_sequence_last_2_str'] = df['fb_sequence_last_2'].map(label_mapping_short)

    ## Statistical test: Spearman correlation within each session, then Wilcoxon signed-rank test of the rhos across sessions
    session_keys = ['monkey', 'session'] if 'monkey' in behav_new.columns else ['session']
    rhos = []
    for _, sess in behav_new.groupby(session_keys):
        if sess['fb_sequence'].nunique() >= 3:  # need at least 3 outcome histories for a correlation
            rhos.append(stats.spearmanr(sess['fb_sequence'], sess['V_t'])[0])
    rhos = np.array(rhos)
    rhos = rhos[~np.isnan(rhos)]
    W, pval, r_rb, n_used = wilcoxon_signed_rank(rhos)
    rho_median = np.median(rhos)

    # previous approach, for comparison only (treats session x history means as independent points)
    rho_pooled, pval_pooled = stats.spearmanr(df['fb_sequence'], df['V_t'])

    print_stat_header(fig_label, behav_new)
    print(f"  Spearman correlation between outcome history (0-7) and neural value within each session,")
    print(f"  then two-sided Wilcoxon signed-rank test of the session rhos against 0")
    print(f"  n = {len(rhos)} sessions ({n_used} non-zero rhos); median rho = {rho_median:.3f} "
          f"(IQR {np.percentile(rhos, 25):.3f} to {np.percentile(rhos, 75):.3f})")
    ci_low, ci_high = bootstrap_median_ci(rhos)
    print(f"  95% CI of the median rho (bootstrap over sessions, 10000 resamples): [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  W = {W:.1f}, p = {pval:.3e}, p (Bonferroni x{n_comparisons}) = {min(pval * n_comparisons, 1):.3e}")
    print(f"  Effect size: matched-pairs rank-biserial r = {r_rb:.3f}")
    print(f"  (old pooled approach, not for reporting: rho = {rho_pooled:.3f}, p = {pval_pooled:.3e}, n = {len(df)} points)")

    # Optional: annotate plot with significance result
    ax.text(0.5, 0.95, f"median rho = {rho_median:.2f}, p = {pval:.1e}",
            ha='center', va='top', transform=ax.transAxes,
            fontsize=6 if paper_format else 10)

    sns.boxplot(df.sort_values('fb_sequence_str'), x='fb_sequence_str', hue='fb_sequence_str', y='V_t', palette=colors, ax=ax, showfliers=showfliers, boxprops={'linewidth': 1}, legend=False, width=0.5)
    if show_datapoints:
        if show_datapoints:
            # Get unique groups and their positions
            unique_groups = df['fb_sequence_str'].unique()
            for i, group in enumerate(sorted(unique_groups)):
                group_data = df[df['fb_sequence_str'] == group]
                x_pos = np.full(len(group_data), i - 0.3)  # Shift left by 0.15
                x_pos += np.random.normal(0, 0.05, size=len(group_data))  # Add jitter
                ax.scatter(x_pos, group_data['V_t'], color=colors[i], s=s, alpha=0.5, edgecolor='none', linewidth=0.5, label=group, zorder=0)

    ax.axhline(0, color='k', linestyle='--', alpha=0.7, linewidth=0.75, zorder=0)

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
    #ax.set_xlabel('Feedback sequence')
    ax.set_ylabel('Neural value')
    # set ticks at every 1
    y_min = df['V_t'].min()
    y_max = df['V_t'].max()
    ax.set_ylim(-np.abs(y_max), np.abs(y_max))
    y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)

    #plt.tight_layout()
    if axis_original is None:
        return fig, ax

def plot_prop_signif_sessions(cpd_temp, ax=None):
    axis_original = ax
    # Initialize container
    plt.rcParams.update({'font.size': 8})

    monkey = cpd_temp.monkey.unique()[0]
    subregion = cpd_temp.subregion.unique()[0]
    
    regressors = ["_".join(col.split('_')[1:]) for col in cpd_temp.columns if col.startswith('cpd_R_')]

    counts = {}
    for reg in regressors:
        #count = (cpd_temp[f'pval_{reg}'] < 0.05).sum()
        count_corrected = (cpd_temp[f'pval_{reg}_corrected'] < 0.05).sum()
        counts[reg] = count_corrected
        #row[f'{reg}_corrected'] = count_corrected

    # small gap
    if axis_original is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))

    n_sessions = len(cpd_temp)

    # Get row from sig_counts_df

    props = np.array([counts[reg] / n_sessions for reg in regressors])

    ax.bar(range(1, len(regressors)+1), props * 100, color=COLORS[subregion], alpha=0.8)
    ax.set_xticks(range(1, len(regressors)+1))
    ax.set_xticklabels(regressors, rotation=45, ha='right')
    #ax.set_ylim(0, session_n + 1)
    ax.set_ylabel('% Significant \nSessions')
    ax.set_title(f'{monkey} {subregion} \n(n={n_sessions})')
    #ax.axhline(y=5, color='black', linestyle='--', alpha=0.3)  # Optional threshold line

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #plt.tight_layout()

    if axis_original is None:
        return fig, ax

def green_red_plot(behav_new, paper_format=False, ax=None):
    from matplotlib.gridspec import GridSpec

    axis_original = ax
    
    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 2  # in cm
        w = 2
        s = 10
        linewidth = 0.7
    else:
        plt.rcParams.update({'font.size': 12})
        h = 6  # in cm
        w = 6  # in cm
        s = 10
        linewidth = 1

    if ax is None:
        # Method 1: Use GridSpec directly (recommended)
        fig, ax = plt.subplots(1, 1, figsize=(w, h))

    # extract datapoints
    V_t = behav_new.V_t.values
    V_t_p1 = behav_new.V_t_p1.values
    fb_vector = behav_new.feedback.values

    V_min, V_max = np.min(V_t), np.max(V_t)
    x_lines = np.arange(V_min, V_max, 0.01)

    V_t_p1_min, V_t_p1_max = np.min(V_t_p1), np.max(V_t_p1)
    # remove outliers (5%)

    axis_extreme = np.max(np.abs([V_t.min(), V_t.max(), V_t_p1.min(), V_t_p1.max()]))

    # Scatter positive and negative feedback with fit lines
    for fb_curr in [0, 1]:
        # get pos/neg trial
        V_t_curr = V_t[fb_vector == fb_curr]
        V_t_p1_curr = V_t_p1[fb_vector == fb_curr]

        # scatter trials
        ax.scatter(V_t_curr, V_t_p1_curr, color=COLORS[fb_curr], alpha=.5, s=s, edgecolor='none')

        # fit and plot slopes
        slope, intercept, r_value, p_value, std_err = stats.linregress(V_t_curr, V_t_p1_curr)
        fit_line = slope * x_lines + intercept
        ax.plot(x_lines, fit_line, color=COLORS[fb_curr], alpha=.7, lw=2, label=f'slope {slope:.2f} p={p_value:.2f}', zorder=10)

        # mean line for each group
        #mean_dV = np.mean(dV_curr)
        #ax_hist.axhline(mean_dV, color=COLORS[fb_curr], linestyle='--', alpha=0.7, linewidth=.7, label=f'mean fb {["negative", "positive"][fb_curr]}')

    # Diagonal line negative slope
    ax.plot(x_lines, x_lines, color='k', alpha=0.7, lw=2, zorder=8)

    # Reference lines
    mean_V_t = 0
    mean_V_t_p1 = 0
    ax.axvline(mean_V_t, color='k', linestyle='--', alpha=0.7, linewidth=linewidth)
    ax.axhline(mean_V_t_p1, color='k', linestyle='--', alpha=0.7, linewidth=linewidth)

    # Add vertical reference lines
    ax.legend(loc='upper left', fontsize=8 if paper_format else 10)

    # Add horizontal reference lines
    #ax_hist.axhline(0, color='k', linestyle='-', alpha=0.7, linewidth=.7)

    # Styling and labels
    ax.scatter([], [], color=COLORS[1], label='Unrewarded')
    ax.scatter([], [], color=COLORS[0], label='Rewarded')

    #ax.legend(loc='upper center')

    ax.set_xlabel('$V_t$', fontsize=10 if paper_format else 12)
    ax.set_ylabel('$V_{t+1}$', fontsize=10 if paper_format else 12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim([-axis_extreme, axis_extreme])
    ax.set_ylim([-axis_extreme, axis_extreme])

    # Create a second axis for the histogram
    if axis_original is None:
        return fig, ax

def plot_simple_glm_weights(weights_all, ax=None, show_data=True):
    axis_original = ax
    plt.rcParams.update({'font.size': 8})

    if ax is None:
        fig, ax = plt.subplots(1, 1 , figsize=(2, 1.5))

    # font size
    weight_cols = [col for col in weights_all.columns if col.startswith('R_')]
    subregion = weights_all.subregion.unique()[0]
    monkey = weights_all.monkey.unique()[0]
    # Calculate min and max

    weights_filtered = weights_all.replace([np.inf, -np.inf], np.nan).dropna()  # Replace inf values with NaN, then drop NaN values
    weights_temp = weights_filtered.drop(columns=['monkey', 'session', 'subregion'])

    ymax = weights_temp.mean().max()
    ymin = weights_temp.mean().min()

    # Visualization - mean weights with error bars and in the background plot individual sessions with lines connecting them
    # weights temp is a DataFrame with weights for each regressor
    weights = weights_temp.mean().values
    weights_err = weights_temp.std().values #/ np.sqrt(len(weights_temp))

    ax.errorbar(range(1, len(weights) + 1), weights, yerr=weights_err,
           fmt='o-', color=COLORS[subregion], 
           markeredgecolor='black', markeredgewidth=.5,
           label='session mean ± SEM', 
           markersize=6, capsize=2)
    # add stirplot
    if not show_data:
        for i, col in enumerate(weights_temp.columns):
            xpos = i+1 -.2
            jitter = np.random.normal(0, 0.05, size=len(weights_temp[col]))
            ax.scatter(xpos + jitter, weights_temp[col].values, color='black', alpha=0.2, zorder=0, s=1)
    else:
        for i, row in weights_temp.iterrows():
            ax.plot(range(1, len(weights) + 1), row.values, color='grey', alpha=0.1)


    # fit exponential and print params
    '''def exponential_fit(x, C, a, b):
        return C*(a * (1-a) ** (x - 1)) + b
    from scipy.optimize import curve_fit
    x_data = np.arange(1, len(weights) + 1)
    try:
        popt, _ = curve_fit(exponential_fit, x_data, weights, p0=(1, 0.5, 0))
        fit_line = exponential_fit(x_data, *popt)
        ax.plot(x_data, fit_line, color=COLORS[subregion], linestyle='--', label=f'{popt}', zorder=1)
    except Exception as e:
        print(f"Error fitting exponential for {monkey} {subregion}: {e}")
        fit_line = None'''



    # Plot the weights as a bar plot

    
    #ax.scatter(range(1, len(weights) + 1), weights, color='steelblue', alpha=0.7, s=80, edgecolor='k')
    ax.axhline(0, color='black', linestyle='dotted', alpha=0.3)
    ax.set_xlabel('Regressor Index')
    ax.set_ylabel('Regression weight')
    ax.set_xticks(range(1, len(weights) + 1))
    ax.set_xticklabels([f'R_{i}' for i in range(1, len(weights)+1)], rotation=45, ha='right')
    #ax.grid(True, alpha=0.3)
    #ax.legend(loc='upper right', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #ax.set_ylim(ymin - 0.1, ymax + 0.1)

    ax.set_title(f'{monkey} {subregion}')

    if axis_original is None:
        return fig, ax

def plot_cpds_history_neural_value(cpd_temp, null=None, ax=None, show_data=False, fig_label='Fig. 4d'):
    axis_original = ax
    if ax is None:
        h = 2
        w = 2.5
        fig, ax = plt.subplots(1, 1, figsize=(w, h))

    monkey, subregion = cpd_temp.monkey.unique()[0], cpd_temp.subregion.unique()[0]

    cpd_cols = [col for col in cpd_temp.columns if col.startswith('cpd_R_')]
    ymax = cpd_temp.groupby(['monkey', 'subregion'])[cpd_cols].mean().max().max()

    cpd_cols = [col for col in cpd_temp.columns if col.startswith('cpd_R_')]
    cpd_vals = cpd_temp[cpd_cols]

    cpd_means = cpd_vals.mean().values
    cpd_sems = cpd_vals.std().values #/ np.sqrt(len(cpd_vals))

    # Plot means + error bars
    ax.errorbar(range(1, len(cpd_means)+1), cpd_means, yerr=cpd_sems,
           fmt='-', color=COLORS[subregion], zorder=2,
           label='session mean ± SEM', 
           markersize=6, capsize=2)
    
    # Plot individual sessions
    # add stirplot
    if show_data:
        # show individual sessions with dots
        for i, col in enumerate(cpd_vals.columns):
            xpos = i+1 -.3
            jitter = np.random.normal(0, 0.05, size=len(cpd_vals[col]))
            ax.scatter(xpos + jitter, cpd_vals[col].values, color='black', alpha=.2, zorder=0, s=2)
        '''# show connected lines
        for _, row in cpd_vals.iterrows():
            ax.plot(range(1, len(cpd_cols)+1), row.values, color='grey', alpha=0.1)'''

    # Stars: compare each regressor to null
    pvals = []
    for i, reg in enumerate(cpd_cols):
        if null is None:  # if there is no null, we assume that there is only one session in the data: print pvalue of CPD
            pcol = f'pval_{"_".join(reg.split("_")[1:])}'
            pval = cpd_temp[pcol].values[0]
        else:
            pval = np.sum(null >= cpd_means[i]) / null.shape[0]
        pvals.append(pval)

    # FDR (Benjamini-Hochberg) correction across lags
    _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

    # print statistics
    print_stat_header(fig_label, cpd_temp)
    print(f"  One-sided permutation test of mean CPD vs. null (CPD of a shuffled regressor, "
          f"{null.shape[0] if null is not None else 'n/a'} permutations), FDR-corrected (Benjamini-Hochberg) across lags")
    print(f"  n = {len(cpd_vals)} sessions; data are mean +/- s.d. across sessions")
    for i, reg in enumerate(cpd_cols):
        print(f"  t-{i+1}: CPD = {cpd_means[i]:.2f} +/- {cpd_sems[i]:.2f} %, p = {pvals[i]:.4f}, p (FDR) = {pvals_fdr[i]:.4f}"
              f"{'  *' if pvals_fdr[i] < 0.05 else ''}")

    # Markers - filled for significant, white for non-significant
    for i in range(len(cpd_means)):
        facecolor = COLORS[subregion] if pvals_fdr[i] < 0.05 else (1, 1, 1, 0.8)
        ax.plot(i+1, cpd_means[i], 'o', markerfacecolor=facecolor,
                markeredgecolor='black', markeredgewidth=.5,
                markersize=6, zorder=3)

    ax.axhline(0, color='black', linestyle='dotted', alpha=0.3)
    ax.set_title(f'{monkey} {subregion}')
    ax.set_ylabel('CPD (%)')
    #ax.set_ylim(-5, ymax+10)
    ax.set_xticks(range(1, len(cpd_cols)+1))
    ax.set_xticklabels([f't-{i}' for i in range(1, len(cpd_cols)+1)], rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.legend(loc='upper right', fontsize=8)

    if axis_original is None:
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



def stat_tests_mixed_glm(results_mixed, alpha=0.05):
    '''
    Extract GLM coeffs and perform statistical tests for the model:
    A = b0 + b1*B + b2*C + b3*(B*C) + error

    where
    A = V_t_p1 (neural value at time t+1)
    B = V_t (neural value at time t)
    C = feedback (0 or 1)
    '''
    ##### Slope and intercept for C=0 and C=1 #####
    b0 = results_mixed.params['const']  
    b1 = results_mixed.params['B']
    b2 = results_mixed.params['C']
    b3 = results_mixed.params['B_C']

    ## Get confidence intervals
    conf_int = results_mixed.conf_int(alpha)

    ##### Slope significant? #####

    # For C=0: slope is b1, intercept is b0
    # For C=1: slope is b1+b3, intercept is b0+b2

    # For C=0: test if b1 != 0
    p_value_b1 = results_mixed.pvalues['B']
    
    # For C=1: slope is b1+b3, need to compute SE and test
    # Using formula: Var(b1 + b3) = Var(b1) + Var(b3) + 2*Cov(b1, b3)
    cov_matrix = results_mixed.cov_params()
    var_b1_plus_b3 = cov_matrix.loc['B', 'B'] + cov_matrix.loc['B_C', 'B_C'] + 2*cov_matrix.loc['B', 'B_C']
    se_b1_plus_b3 = np.sqrt(var_b1_plus_b3)
    t_stat_c1 = (b1 + b3) / se_b1_plus_b3
    p_value_c1 = 2 * (1 - stats.t.cdf(abs(t_stat_c1), df=results_mixed.df_resid))
    


    ##### Slope between 0 and 1? #####
    # For C=0: check if confidence interval for b1 is within [0, 1]
    ci_b1_lower = conf_int.loc['B', 0]
    ci_b1_upper = conf_int.loc['B', 1]
    within_bounds_c0 = (ci_b1_lower >= 0) and (ci_b1_upper <= 1)

    # For C=1: confidence interval for b1+b3
    ci_b1_plus_b3_lower = (b1 + b3) - 1.96 * se_b1_plus_b3
    ci_b1_plus_b3_upper = (b1 + b3) + 1.96 * se_b1_plus_b3
    within_bounds_c1 = (ci_b1_plus_b3_lower >= 0) and (ci_b1_plus_b3_upper <= 1)
    


    ##### Test if intercept is higher for C=1 vs C=0 #####

    # Intercept for C=0 is b0
    # Intercept for C=1 is b0 + b2
    # Test if (b0 + b2) > b0, i.e., if b2 > 0
    p_value_b2 = results_mixed.pvalues['C']
    t_stat_b2 = results_mixed.tvalues['C']

    # One-sided test: H0: b2 <= 0, H1: b2 > 0
    if t_stat_b2 > 0:
        p_value_one_sided = p_value_b2 / 2  # One-sided
    else:
        p_value_one_sided = 1 - p_value_b2 / 2

    ##### Report results #####

    return {
        'slope_c0': b1,  # slope of V_t for R=0 (unrewarded)
        'slope_c1': b1 + b3,  # slope of V_t for R=1 (rewarded)
        'intercept_c0': b0,  # intercept for R=0 (unrewarded)
        'intercept_c1': b0 + b2,  # intercept for R=1 (rewarded)

        'b1_c0_pvalue': p_value_b1,  # p-value for slope of V_t for R=0 (unrewarded)
        'b1_c0_significant': p_value_b1 < 0.05,  # is slope of V_t for R=0 (unrewarded) significant?
        'b1_c1_pvalue': p_value_c1,  # p-value for slope of V_t for R=1 (rewarded)
        'b1_c1_significant': p_value_c1 < 0.05,  # is slope of V_t for R=1 (rewarded) significant?

        'b1_c0_ci': (ci_b1_lower, ci_b1_upper),  # confidence interval for slope of V_t for R=0 (unrewarded)
        'b1_c0_within_bounds': within_bounds_c0,  # is slope of V_t for R=0 (unrewarded) between 0 and 1?
        'b1_c1_ci': (ci_b1_plus_b3_lower, ci_b1_plus_b3_upper),  # confidence interval for slope of V_t for R=1 (rewarded)
        'b1_c1_within_bounds': within_bounds_c1,  # is slope of V_t for R=1 (rewarded) between 0 and 1?

        'intercept_difference': b2,
        'intercept_c1_higher': p_value_one_sided < 0.05  # is intercept for R=1 (rewarded) significantly higher than intercept for R=0 (unrewarded)?
    }


def fit_mixed_glm(data):
    '''
    Fits the model :

    A = b0 + b1*B + b2*C + b3*(B*C) + error

    on the data. Data needs to have columns 'A', 'B', and 'C'.
    '''
    # Create design matrix with interaction term
    X = pd.DataFrame({
        'const': 1,
        'B': data['B'],
        'C': data['C'],
        'B_C': data['B'] * data['C']  # Interaction term
    })

    # Fit the mixed GLM
    model_mixed = sm.OLS(data['A'], X)
    results_mixed = model_mixed.fit()

    return results_mixed


def add_hypothesis(test_results):
    '''
    Returns a hypothesis label based on the test results. 
    
    Hypotheses:
    Delta rule: both slopes are within bounds and intercept is higher for C=1
    Feedback: slopes are not significant but intercept is higher for C=1
    Nothing: no slopes significant and no intercept difference
    RPE?: slopes are significant but not within bounds, and intercept is higher for C=1
    else: does not fit any of the above categories
    '''
    if (test_results['b1_c0_within_bounds'] and 
        test_results['b1_c1_within_bounds'] and
        test_results['b1_c0_significant'] and
        test_results['b1_c1_significant'] and  # if both slopes are within bounds (0, 1) and intercet positive is higher
        test_results['intercept_c1_higher']):
        return 'Delta rule (strict)'
    elif ((test_results['b1_c0_within_bounds'] or test_results['b1_c1_within_bounds']) and
          test_results['intercept_c1_higher']):  # if any slopes are within bounds (0, 1) and intercet positive is higher
        return 'Delta rule (loose)'
    elif (not test_results['b1_c0_significant'] and not test_results['b1_c1_significant'] and  # if no slope is significantly nonzero but there is a difference
          test_results['intercept_c1_higher']):
        return 'Feedback'
        '''elif (not test_results['b1_c0_significant'] and not test_results['b1_c1_significant'] and not test_results['intercept_c1_higher']):return 'Nothing'''
    else:
        return 'Other'

