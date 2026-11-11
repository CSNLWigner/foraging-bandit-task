import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, permutation_test_score, KFold
import xarray as xr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr


from popy.io_tools import load_behavior, load_neural_data
from popy.neural_data_tools import *
from popy.behavior_data_tools import *
from popy.config import PROJECT_PATH_LOCAL


# Data processing #
def _get_data_of_interest(neural_data, step_len):
    '''
    when we do not want to decode every time point, we can subsample the data to only include the time points of interest. This function does that.
    '''
    time_original = neural_data.time.data
    times_of_interest = np.arange(time_original[0], time_original[-1], step_len)
    # find times closest to times of interest
    times_idx = [np.argmin(np.abs(time_original - time)) for time in times_of_interest]
    times_of_interest = time_original[times_idx]
    return neural_data.isel(time=times_idx)


def _create_results_xr(conditions, group_targets, time_vector, areas):
    coords = {
            'target': np.array(conditions, dtype='U50'),  # Unicode string with max length 50
            'area': np.array(areas, dtype='U10'),         # Unicode string with max length 10
            'time': time_vector,
        }
    
    shape = (len(conditions), len(group_targets), len(time_vector), len(areas))
    data_vars = {
        'scores': (['target', 'group_target', 'time', 'area'], np.full(shape, np.nan)),
        'pvals': (['target', 'group_target', 'time', 'area'], np.full(shape, np.nan)),
        'perm_mean': (['target', 'group_target', 'time', 'area'], np.full(shape, np.nan)),
        'perm_std': (['target', 'group_target', 'time', 'area'], np.full(shape, np.nan)),
        #'cv_mean': (['target', 'time', 'area'], np.full((len(conditions), len(time_vector), len(areas)), np.nan)),
        #'cv_std': (['target', 'time', 'area'], np.full((len(conditions), len(time_vector), len(areas)), np.nan)),
    }

    coords['group_target'] = np.array(group_targets, dtype='U50')  # Unicode string with max length 50

    return  xr.Dataset(data_vars=data_vars, coords=coords)


def _label_encoder(y):
    """
    Converts a vector of labels to numerical values.
    """
    # if there are nans
    assert not np.isnan(y).any(), "Label vector contains NaN values."

    # encode as integers
    le = LabelEncoder()
    le.fit(y)
    y_numerical = le.transform(y)

    return y_numerical
    
    """
    This part replaced nan values with nan values in the labels, but it is not necessary anymore since we remove nan trials in the first place.

    # array of zeros, set dtype to , dtype=object
    y_new = np.zeros(y_numerical.shape, dtype='float64')

    # reset values to nan if they were nan (LabelEncoder sets nan to some numerical value as well...)
    # Also, for some reason, labels are relabeled to start from 0...
    class_names = le.classes_ # get the class names
    for i, class_name in enumerate(class_names):
        if class_name == 'nan':
            y_new[np.where(y_numerical == i)] = np.nan
            print("nan s'est glisse dans les labels")
        else:
            y_new[np.where(y_numerical == i)] = i

    return y_new"""


def _preproc_data(neural_dataset_original, 
                  target_name, 
                  group_target_name=None
                  ):
    """
    Creates a matrix of regressors (X) and a vector of labels (y) from the neural data.
    
    Includes 3 steps:
    0. Creating data matrix and labels
    1. Remove nan trials
    2. If categorical, create numerical labels and balance the dataset
    
    Returns the data matrix and labels, ready to plug into the decoder.
    """

    neural_dataset = neural_dataset_original.copy(deep=True)

    # Step 0. Drop nan trials from labels (and groups)
    neural_dataset = remove_nan_labels(neural_dataset, target_name)  # drop trials with nan behavioural label
    if group_target_name is not None:
        neural_dataset = remove_nan_labels(neural_dataset, group_target_name)  
    
    # Step 1. Balance classes
    #X, y = _balance_categorical(X, y)
    n_classes = len(np.unique(neural_dataset[target_name].data))
    
    n_trials_before = len(neural_dataset)
    coords = None  # coords to balance the dataset on
    if n_classes <= 5 and group_target_name is None:  # if y is categorical and groups are not provided
        coords = target_name
    elif n_classes <= 5 and group_target_name is not None:  # if y is categorical and groups are provided
        coords = [target_name, group_target_name]
    elif n_classes > 5 and group_target_name is not None:  # if y is continuous and groups are provided
        coords = group_target_name

    if coords is not None:  
        neural_dataset = balance_labels(neural_dataset, coords=coords)  # TODO: uncomment this line after target CV decoder!!!!
        n_trials_after = len(neural_dataset)
        #print(f"Removed {n_trials_before - n_trials_after}/{n_trials_before} to balance the dataset on: {coords}")
    #else:
        #print(f"Dataset is not balanced, no trials removed.")

    # Step 2. Creating data matrix and labels
    X = neural_dataset.data
    y = neural_dataset[target_name].data
    if group_target_name is not None:
        y_groups = neural_dataset[group_target_name].data
    else:
        y_groups = None

    # Step 3. Encode categoricaxl labels
    if len(np.unique(y)) <= 5:  # if y is categorical
        y = _label_encoder(y)  # relabel from 0

    if y_groups is not None:  # if groups are provided, encode them as well
        y_groups = _label_encoder(y_groups)

    return X, y, y_groups

# Decoding #

# Create a custom scorer that returns the Pearson correlation coefficient
def pearson_scorer(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]  # Returns the correlation value, not the p-value

# train on one target, test on the others
def target_cv_score(decoder, X_temp, y, groups):
    scores_all = []
    for group_temp in np.unique(groups):
        # train test split: train on one target, test on the others
        train_ids, test_ids = np.where(groups == group_temp)[0], np.where(groups != group_temp)[0]  # train on one target, test on the others
        y_train = y[train_ids]  # y_train is the target variable for the training set
        y_test = y[test_ids]  # y_test is the target variable for the test set
        X_temp_train = X_temp[train_ids, :]  # X_temp_train is the neural data for the training set
        X_temp_test = X_temp[test_ids, :]  # X_temp_test is the neural data for the test set

        # fit the decoder on the training data, evaluate on the test data
        decoder_fit = decoder.fit(X_temp_train, y_train)  # fit the decoder on the training data
        y_pred = decoder_fit.predict(X_temp_test)  # predict the test data
        corr = pearson_scorer(y_test, y_pred)  # compute the correlation between the predicted and true values
        scores_all.append(corr)  # append the correlation to the list

    # compute the mean correlation across all targets
    score = np.mean(scores_all)  # take the mean of the correlations
    return score


def linear_decoding(X, y, groups=None, K_fold=None, n_perm=1000, n_jobs=-1):
    """
    Decoding the target variable 'y' from the neural data matrix 'X'.

    In case of continuous data, the decoder is a linear regression, and the scoring measure is 'r2'.
    In case of discrete data, the decoder is a logistic regression, and the scoring measure is 'accuracy'.
    """


    ## DECODER SETUP
    if K_fold is None:
        K_fold = 10

    # is data continuous or discrete?
    dtype = 'continuous' if len(np.unique(y)) > 5 else 'discrete'
    if dtype == 'continuous':  # in case of continuous data -> regression
        if groups is None:
            kf = KFold(n_splits=K_fold, shuffle=True)
        else:
            kf = LeaveOneGroupOut()  # use LeaveOneGroupOut decode the value across targets (trained on one target, tested on the other) 
        #decoder = LinearRegression() 
        alphas = np.logspace(-1, 4, 20)  # alphas for the RidgeCV decoder, 
        decoder = Pipeline([
            ('scaler', StandardScaler()),
            ('decoder', Ridge(alpha=100))
        ])
        #decoder = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=kf)
        scoring_function = 'r2'
        # Create a scorer object
        #scoring_function = make_scorer(pearson_scorer)  # TODO this is custom for the across target CV, change this when finished

    elif dtype == 'discrete':  # in case of dicrete data -> classification
        if groups is None:
            kf = StratifiedKFold(n_splits=K_fold, shuffle=True)  # use stratified KFold to preserve the ratio of classes in each fold
        else:
            kf = LeaveOneGroupOut()
            #kf = LeaveOneGroupOut()
        decoder = LogisticRegression()
        scoring_function = 'accuracy'

    ## DECODING LOOP

    # scores of shape (K_fold, n_bins)
    scores = np.empty(X.shape[2])
    perm_scores_mean = np.empty(X.shape[2])
    perm_scores_std = np.empty(X.shape[2])
    pvals = np.empty(X.shape[2])
    #cv_mean_scores = np.empty(X.shape[2])
    #cv_std_scores = np.empty(X.shape[2])

    # loop through time points
    for t_idx in np.arange(X.shape[2]):
        # print progress

        X_temp = X[:, :, t_idx]  # select (trial, unit) data at time t_idx
        if groups is None:
            if n_perm is None:
                cv_scores = cross_val_score(decoder, X_temp, y, cv=kf, scoring=scoring_function, n_jobs=n_jobs)
                score = cv_scores.mean()  # take the mean of the cross-validation scores
                pvalue = np.nan  # if no permutation test is requested, set pvalue to nan
                perm_scores_mean_temp = np.nan
                perm_scores_std_temp = np.nan

            elif n_perm is not None:
                score, perm_scores, p_value = permutation_test_score(
                    decoder, X_temp, y, cv=kf, scoring=scoring_function,
                    n_permutations=n_perm, n_jobs=n_jobs,
                )
                
                # Calculate p-value
                pvalue = p_value  # p-value from the permutation test
                perm_scores_mean_temp = perm_scores.mean()  #
                perm_scores_std_temp = perm_scores.std()  # compute                

        elif groups is not None and n_perm is not None:
            score = target_cv_score(decoder, X_temp, y, groups)  # compute the score for the current time point

            # compute the permutation test
            # shuffle the labels and compute the score

            perm_scores_all = []
            for i in range(n_perm):
                # shuffle the labels
                y_temp = np.random.permutation(y)
                # compute the score
                score_temp = target_cv_score(decoder, X_temp, y_temp, groups)
                perm_scores_all.append(score_temp)
            perm_scores_mean_temp = np.mean(perm_scores_all)  # convert to numpy array
            perm_scores_std_temp = np.std(perm_scores_all)  # compute the std of the permutation scores

            pvalue = np.sum(np.abs(perm_scores_all) >= np.abs(score)) / n_perm  # compute the pvalue

        # store the results
        scores[t_idx] = score
        pvals[t_idx] = pvalue
        perm_scores_mean[t_idx] = perm_scores_mean_temp
        perm_scores_std[t_idx] = perm_scores_std_temp

        # cross-validation scores without permutation (sanity check)
        '''score = cross_val_score(decoder, X_temp, y, groups=groups, cv=kf, scoring=scoring_function)
        cv_mean_scores[t_idx] = np.mean(score)
        cv_std_scores[t_idx] = np.std(score)'''

    return scores, pvals, perm_scores_mean, perm_scores_std#, cv_mean_scores, cv_std_scores


def load_data_for_decoder(monkey, session, n_extra_trials=(-1, 1)):
    # 1. Behavior data

    # Load behavior data
    behav = load_behavior(monkey, session)
    behav = drop_time_fields(behav)

    # add behav vars to decode: stay value with different learning rates (Fig. 4g) and with the fitted one (Fig. 3f)
    for alpha in np.arange(0.05, 1, 0.05):
        behav = add_foraging_value(behav, alpha=alpha)
        behav = behav.rename(columns={'stay_value': f'stay_value_{alpha:.2f}'})
    behav = add_foraging_value(behav)  # add shift value for its decoding

    # Q values of the inferential and standard RL models (Supp. Fig. 6), as in git 86e1b5c
    behav = add_RL_values(behav, structure_aware=True)  # add RL values for its decoding
    # rename Q_1 to Q_1_inf, Q_2 to Q_2_inf, Q_3 to Q_3_inf for the inferential model
    behav = behav.rename(columns={'Q_1': 'Q_1_inf', 'Q_2': 'Q_2_inf', 'Q_3': 'Q_3_inf'})
    # add a new col Q_chosen_inf, which is the Q value of the chosen option (target column)
    behav['Q_chosen_inf'] = behav.apply(lambda row: row[f'Q_{int(row["target"])}_inf'] if not np.isnan(row["target"]) else np.nan, axis=1)

    behav = add_RL_values(behav, structure_aware=False)  # add RL values for its decoding
    behav = behav.rename(columns={'Q_1': 'Q_1_stand', 'Q_2': 'Q_2_stand', 'Q_3': 'Q_3_stand'})
    behav['Q_chosen_stand'] = behav.apply(lambda row: row[f'Q_{int(row["target"])}_stand'] if not np.isnan(row["target"]) else np.nan, axis=1)
    
    # 2. Neural data

    # Load neural data
    neural_data = load_neural_data(monkey, session, hz=1000)
    n_units_all = len(neural_data['unit'].values)

    # remove some units
    neural_data = remove_low_fr_neurons(neural_data, 1, print_usr_msg=False)
    neural_data = remove_trunctuated_neurons(neural_data, mode='remove', delay_limit=10, print_usr_msg=False)
    n_units_kept = len(neural_data['unit'].values)
    if len(neural_data['unit'].values) == 0:
        raise ValueError(f"No neurons left for {monkey}_{session}")
    
    # process neural data
    neural_data = add_firing_rates(neural_data, drop_spike_trains=True, method='gauss', std=.05)
    neural_data = downsample_time(neural_data, 100)
    neural_data = scale_neural_data(neural_data)
    neural_data = time_normalize_session(neural_data)

    # 3. build neural dataset and merge with behavior
    neural_dataset = build_trial_dataset(neural_data, mode='full_trial', n_extra_trials=n_extra_trials)
    neural_dataset = merge_behavior(neural_dataset, behav)

    print(f"Monkey: {monkey}, Session: {session} - Removed {n_units_all - n_units_kept} / {n_units_all} neurons")

    return neural_dataset


### Decoding pipeline

def run_decoder(monkey, session, PARAMS, n_jobs=1, load_data=False, save_data=False):
    """
    Decoding the target variable from the neural data.

    Includes these steps:
    1. loads data, preprocess it, create labelled dataset
    2. run decoder on the labelled dataset

    Returns the scores of the decoder.

    """
    # if targets is not a list, make it a list (i.e. if only one target is passed as string)
    if not isinstance(PARAMS['conditions'], list): 
        PARAMS['conditions'] = [PARAMS['conditions']]
    if not isinstance(PARAMS['group_targets'], list):
        PARAMS['group_targets'] = [PARAMS['group_targets']]

    # print log
    print(f"Running for monkey {monkey} and session {session}")

    ## init data
    #load if exist, else prepare
    floc = os.path.join(PROJECT_PATH_LOCAL, 'notebooks', 'population_decoding', 'data', f'{monkey}_{session}_neural.nc')
    if os.path.exists(floc) and load_data:
        neural_dataset = xr.open_dataset(floc)
    else:
        neural_dataset = load_data_for_decoder(monkey, session, PARAMS['n_extra_trials'])

    if save_data:    
        # save data
        neural_dataset.to_netcdf(floc)
    
    # 0. get bins of interest
    neural_data = _get_data_of_interest(neural_dataset, PARAMS['step_len'])

    # Create an empty dataset with dimensions [target, *, time, area] - groups are added to * if its not None, else this dimension is omitted
    xr_scores = _create_results_xr(
        PARAMS['conditions'], 
        PARAMS['group_targets'],
        time_vector= neural_data.time.data,
        areas=['LPFC', 'MCC'])

    for target in PARAMS['conditions']:
        for group_target in PARAMS['group_targets']:
            # sanity: groups must be categorical
            if group_target is not None:
                assert len(np.unique(neural_data[group_target].data)) <= 5, f"Groups must be categorical - {group_target}"

            for area in np.unique(neural_data.area.values):
                print(f"Decoding {target} in {area}")
                #try:
                neural_data_temp = neural_data.where(neural_data.area == area, drop=True)

                '''if monkey == 'po' and group_target.split('_')[0] == 'target':
                    # remove trials with target 2 (po_2) for po monkey
                    neural_data_temp = neural_data_temp.where(neural_data_temp.target != 2, drop=True)'''

                if len(neural_data_temp.unit) == 0:
                    continue

                # 1. preprocess data -> create a matrix of regressors (X) and a vector of labels (y)

                X, y, groups = _preproc_data(neural_data_temp.firing_rates, target, group_target)

                # 2. run decoder on the labelled dataset
                scores, pvals, perm_mean, perm_std = linear_decoding(X, y, groups, K_fold=PARAMS['K_fold'], n_perm=PARAMS['n_perm'], n_jobs=n_jobs)

                # create xarray dataset, dimension is time, vars are scores
                xr_scores.scores.loc[target, str(group_target), :, area] = scores
                xr_scores.pvals.loc[target, str(str(group_target)), :, area] = pvals
                xr_scores.perm_mean.loc[target, str(group_target), :, area] = perm_mean
                xr_scores.perm_std.loc[target, str(group_target), :, area] = perm_std
                #xr_scores.cv_mean.loc[target, :, area] = cv_mean
                #xr_scores.cv_std.loc[target, :, area] = cv_std

                #except Exception as e:
                    #print(f"Error decoding {target} in {area}: {e}")

    # add monkey and session information
    xr_scores = xr_scores.assign_coords(session=f'{monkey}_{session}')
    xr_scores = xr_scores.expand_dims('session')
    xr_scores = xr_scores.assign_coords(monkey=('session', [monkey]))
    for key, value in PARAMS.items():
        xr_scores.attrs[key] = str(value)

    session_log=[]  # Log messages of the internal run, not implemented yet
    return xr_scores, session_log     
