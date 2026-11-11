
import pandas as pd
import numpy as np
import statsmodels.api as sm
import xarray as xr
import warnings

from scipy.stats import permutation_test
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn import linear_model
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import PoissonRegressor

from popy.io_tools import load_behavior, load_neural_data
from popy.behavior_data_tools import add_foraging_value, add_switch_info, add_history_of_feedback
from popy.neural_data_tools import *

def fit_eval_glm(y_true, X):
    model = sm.GLM(y_true, X, family=sm.families.Poisson()).fit()
    
    # get coefficients (z-values)
    coeffs = model.params
    tvals = model.tvalues
    pvals = model.pvalues
    
    # predict
    y_pred = model.predict(X)

    # (Pseudo R2, whatever it means) get the explained variance: 1 - (D / total devience), where D is the devience of the model, and total devience is the devience of the mean model
    D_true = mean_poisson_deviance(y_true, y_pred)
    D_mean = mean_poisson_deviance(y_true, np.asarray(y_true).mean() * np.ones_like(y_true))
    ED = (1 - D_true / D_mean) * 100 # get the coefficient of determination (pseudo r2)?

    """
    Same as (more proper according to claude):
    null_deviance = model.null_deviance
    residual_deviance = model.deviance
    pseudo_r2 = (1 - residual_deviance / null_deviance) * 100
    """


    return ED, coeffs, tvals, pvals

def get_CPD(y_true, X_full, X_reduced): 
    # estimate D2 score for the model (full vs reduced)
    model_full = sm.GLM(y_true, X_full, family=sm.families.Poisson()).fit()
    model_reduced = sm.GLM(y_true, X_reduced, family=sm.families.Poisson()).fit()

    # predict
    y_pred_full = model_full.predict(X_full)
    y_pred_reduced = model_reduced.predict(X_reduced)

    # Calculate the mean Poisson deviance for both models (extension of the SSE=mean(yi - yi_pred) for poisson)
    D_reduced = mean_poisson_deviance(y_true, y_pred_reduced)
    D_full = mean_poisson_deviance(y_true, y_pred_full)

    # Avoid division by zero
    if D_reduced == 0:
        return 0.0
    
    CPD = (1 - D_full / D_reduced) * 100

    """    
    Same as (more proper according to claude):
    null_deviance = model_full.null_deviance
    residual_deviance = model_full.deviance
    pseudo_r2 = (1 - residual_deviance / null_deviance) * 100
    """

    return CPD


class SingleUnitAnalysis:
    def __init__(self, monkey, session, area=None):
        """
        endog: count data, 1D array, shape (n_trials)
        exog: predictors, 2D array, shape (n_trials, n_predictors)
        """
        self.step_time = .100  # in seconds, how much we want to slide the glm window
        self.n_permutations = 1000  # number of permutations for the permutation test

        self.monkey = monkey
        self.session = session
        self.area = area  # area of interest, e.g. 'MCC', 'ofc', 'LPFC'

        # glm parameters
        #self.linear_predictors = None  # condition of interest, e.g. ['target', 'feedback', 'value_function]
        #self.anova_predictors = None  # condition of interest, e.g. ['target', 'feedback', 'value_function]
        self.glm_predictors = [None]  # condition of interest, e.g. ['target', 'feedback', 'value_function]
        self.cpd_targets = [None]  # the predictors to remove (one by one) from the full model to get the CPD, e.g. ['feedback', 'value_function'] -> it means two CPD values will be calculated (using 2 models), one for feedback and one for value_function
                
        self.model = None  # model to run, 'linear_correlation' or 'anova' or 'glm' or 'glm_cpd' (only glm and glm_cpd are implemented)
        
        self.neural_data_type = None  # type of neural data, 'spike_counts' or 'firing_rates'
        self.value_type = None  # type of value, 'discrete' or 'continuous'
        self.time_window = [-3, 3]  # time window around event to consider (in seconds)
        #self.n_extra_trials = -1  # number of extra trials to add to the dataset
        
        ###self.glm_results = {'coeffs': None, 'tvals': None, 'p_vals': None}

        self.results = None  # save results
        self.log = []  # write log info in human readable format, e.g. errors, etc.


    def _get_data_of_interest(
        self,
        neural_data: xr.Dataset,
        step_len: float,
    ) -> xr.Dataset:
        '''
        when we do not want to decode every time point, we can subsample the data to only include the time points of interest. This function does that.
        '''
        time_original = neural_data.time.data
        times_of_interest = np.arange(time_original[0], time_original[-1], step_len)
        # find times closest to times of interest
        times_idx = [np.argmin(np.abs(time_original - time)) for time in times_of_interest]
        times_of_interest = time_original[times_idx]
        return neural_data.isel(time=times_idx)


    def _create_results_xr(self, unit_ids, time_vector, scoring):
        # create xarray with unit_ids as unit dimension and time_vector as time dimension
        attrs = self.__dict__.copy()
        # remove 'results', 'monkey', 'session' keys from the attributes...
        for key in ['results', 'monkey', 'session', 'log']:
            if key in attrs.keys():
                attrs.pop(key)
            
        data_xr = xr.Dataset(
            data_vars=dict(
                scores=(['unit', 'time'], scoring),
            ),
            coords=dict(
                unit= [f'{self.monkey}_{self.session}_{unit_id}' for unit_id in unit_ids],  # save unit ids
                time=time_vector,
                monkey=("unit", [self.monkey for _ in range(len(unit_ids))]),
                session=("unit", [self.session for _ in range(len(unit_ids))]),
                area=("unit", [unit_id.split('_')[0] for unit_id in unit_ids]),
                channel=("unit", [unit_id.split('_')[1] for unit_id in unit_ids]),
            ),
            attrs=attrs
        )
        
        for key, value in data_xr.attrs.items():
            data_xr.attrs[key] = str(value)
        
        return data_xr
    

    def permutation_glm(self, y, X):
        raise NotImplementedError("This function is not implemented yet. Please use the permutation_glm_CPD function instead.")
        # Fit the true model
        ED, coeffs, tvals, pvals = fit_eval_glm(y, X)
        
        # Run permutation test - fit shuffled models N times
        n_perm = self.n_permutations
        EDs_perm = np.zeros(n_perm)
        for i in range(n_perm):
            y_perm = np.random.permutation(y)
            ED_perm, coeffs_perm, tvals_perm, pvals_perm = fit_eval_glm(y_perm, X)
            EDs_perm[i] = ED_perm

        # get p-values for the coefficients
        coeffs_pvals = np.zeros(len(coeffs))

            
        return ED, coeffs, tvals, pvals
    

    def permutation_glm_CPD(self, y_true, X_full, target_predictors):
        """
        Computes the CPD score of the target predictors in the full model.

        Parameters
        ----------
        y_true : array
            The dependent variable, e.g. spike counts.
        X_full : array
            The independent variables, e.g. predictors - that makes the FULL model.
        target_predictors : list
            The predictors to remove from the full model to get the CPD score (one by one).

        Returns
        -------
        CPDs_true : array
            The CPD values of the target predictors.
        CPDs_pvals : array
            The p-values of the CPD values.
        """

        CPDs_true = []  # store the true coefficients for each target predictor
        CPDs_pvals = []  # store the p-values for each target predictor
        
        for target_predictor in target_predictors:
            # Prepare data 
            X_reduced = X_full.drop(target_predictor, axis=1)  # drop target predictor from the full model
            
            # Fit the true model, i.e. get cpd for the target predictor
            CPD_true = get_CPD(y_true, X_full, X_reduced)

            # If the CPD is NaN, skip this predictor
            if np.isnan(CPD_true):
                CPDs_true.append(np.nan)
                CPDs_pvals.append(np.nan)
                continue
            
            # Permutation test - fit shuffled models N times
            n_perm = self.n_permutations
            CPD_perm = np.zeros(n_perm)

            for i in range(n_perm):
                # Create permuted version of the full predictor matrix
                X_full_perm = X_full.copy()

                # Permute only the target predictor while preserving index alignment (??)
                X_full_perm[target_predictor] = np.random.permutation(X_full[target_predictor].values)
                ###X_full_perm[target_predictor] = np.random.permutation(X_full[target_predictor])

                CPD_perm[i] = get_CPD(y_true, X_full_perm, X_reduced)
                

            # Calculate p-value (one-tailed test - CPD should be positive)
            # Count how many permuted CPDs are >= true CPD
            p_val = np.mean(CPD_perm >= CPD_true)
            
            CPDs_true.append(CPD_true)
            CPDs_pvals.append(p_val)
        
        return CPDs_true, CPDs_pvals
    
    
    def load_data_for_glm(self):
        """
        First load the all-session behav info.
            - X: behav info, shape of (n_trials)
            - y: spike count data, shape of (n_trials, n_unis, n_timebins)

        Then create dataset.
        """
        
        # 1. Behavior data
        session_data = load_behavior(self.monkey, self.session)
        if not self.value_type is None:
            session_data = add_foraging_value(session_data, digitize=False)
            session_data['value_fb_pos'] = session_data['stay_value'] * session_data['feedback']  # add value feedback interaction
            session_data['value_fb_neg'] = session_data['stay_value'] * (1 - session_data['feedback'])  # add value feedback interaction
        session_data = add_switch_info(session_data)
        session_data = add_history_of_feedback(session_data, num_trials=8, one_column=False)
        session_data = session_data.dropna()  # remove nan values, the glm can not handle them

        # set datatypes

        # 2. Neural data
        neural_data = load_neural_data(self.monkey, self.session, hz=1000)

        # filter neurons by area
        if self.area is not None:
            neural_data = neural_data.sel(unit=neural_data['area'] == self.area)
        
        # neural dataset preprocessing
        neural_data = remove_trunctuated_neurons(neural_data, mode='set_nan', delay_limit=20)  # set off-recording neurons to nan
        # todo: high dispersion neurons should be handled here!!!
        #neural_data = remove_low_varance_neurons(neural_data, var_limit=.2, print_usr_msg=True)  # remove neurons with low variance
        #neural_data = remove_drift_neurons(neural_data, corr_limit=.2)  # remove neurons with drift
        #neural_data = remove_low_fr_neurons(neural_data, 1)  # remove low_firing units
        if self.neural_data_type == 'spike_counts':
            neural_data = add_firing_rates(neural_data, method='count', win_len=0.200, drop_spike_trains=True)  # convert neural data to firing rates or binned spike counts
        elif self.neural_data_type == 'firing_rates':
            neural_data = add_firing_rates(neural_data, method='gauss', drop_spike_trains=True)
        neural_data = downsample_time(neural_data, 100)  # downsample neural data to 100 Hz
        
        # build dataset
        if self.neural_data_type == 'firing_rates':
            neural_data = time_normalize_session(neural_data)
            neural_dataset = build_trial_dataset(neural_data, mode='full_trial', n_extra_trials=self.n_extra_trials)
            neural_dataset = merge_behavior(neural_dataset, session_data)
        elif self.neural_data_type == 'spike_counts':
            # center on feedback pm 2sec
            neural_dataset = build_trial_dataset(neural_data, mode='centering', center_on_epoch_start=5, center_window=self.time_window)  # center on feedback, 2 sec before and after (we can not time normalize spikes thus we can not use the full trial mode)
            neural_dataset = merge_behavior(neural_dataset, session_data)

        return neural_dataset  # save predictors

    ### GLM pipeline

    def run(self, print_log=False):
        # print log
        print(f'Running {self.model} for {self.monkey} - {self.session}...')
        
        ## init data - the loading method should be customized in a child class to return the dataset and predictors of interest - but in a general way
        neural_dataset = self.load_data_for_glm()  # load data, inits self.neural_dataset and self.predictors
        neural_dataset = self._get_data_of_interest(neural_data=neural_dataset, step_len=self.step_time)  # subsample the data to only include the time points of interest

        # time vector, considering the step time
        #time_vector = self._get_time_vector(neural_dataset)
        time_vector = neural_dataset.time.data

        # init results container
        n_units = len(neural_dataset.unit)
        n_time = len(time_vector)
        n_predictors_glm = len(self.glm_predictors)+1  # +1 for the constant term in the glm model

        scoring = np.full((n_units, n_time), np.nan)  # store the explained deviance

        coeffs = np.full((n_units, n_time, n_predictors_glm), np.nan)  # store the coefficients
        tvals = np.full((n_units, n_time, n_predictors_glm), np.nan)  # store the t-values
        coeffs_pvals = np.full((n_units, n_time, n_predictors_glm), np.nan)  # store the p-values of the coefficients

        if self.model == 'glm_cpd':
            n_predictors_CPD = len(self.cpd_targets)  # for the number of target predictors in the CPD model
            CPDs = np.full((n_units, n_time, n_predictors_CPD), np.nan)
            CPDs_pvals = np.full((n_units, n_time, n_predictors_CPD), np.nan)
        
        for unit_id, unit_name in enumerate(neural_dataset.unit.data):  # loop over units
            # get neural data for the given unit
            neural_dataset_unit = neural_dataset[self.neural_data_type].sel(unit=unit_name)
            
            for t, time in enumerate(neural_dataset_unit.time.data):  # loop over time bins
                # get neural data for the given unit and time bin
                neural_dataset_temp = neural_dataset_unit.sel(time=time)  # get spike counts (or firing rates) for given unit and time bin
                neural_dataset_temp = neural_dataset_temp.dropna(dim='trial_id')  # remove nan values from the dataset 
                y = neural_dataset_temp.data # get spike counts (or firing rates) for given unit and time bin

                # get predictors for the given unit and time bin
                # TODO: maybe it is the same in each time bin, so we can do this step outside the loop once
                predictors = neural_dataset_temp.trial_id.to_dataframe() 
                predictors = predictors[self.glm_predictors].copy()  # select predictors of interest
                predictors = sm.add_constant(predictors)  # add constant to the full model

                # skip if all the values are the same in the neural data TODO: here it should rather be a check fo poisson distribution!!!
                if len(np.unique(y)) < 2:
                    self.log.append(f'Skipping {unit_name} at time {time}, all values are the same.')
                    continue

                # Measure 1: fit glm to get the explained deviance, coefficients, t-values, and p-values                    
                if self.model == 'glm' or self.model == 'glm_cpd':
                    # Measure 3: fit glm and run permutation test
                    try:
                        #ED, coeffs_temp, tvals, pvals_temp = self.permutation_glm(y, predictors)  # run permutation test
                        pseudo_r2, coeffs_temp, tvals_temp, pvals_temp = fit_eval_glm(y, predictors)

                        scoring[unit_id, t] = pseudo_r2
                        #p_vals[unit_id, t] = ED_pval
                        coeffs[unit_id, t, :] = coeffs_temp.values
                        tvals[unit_id, t, :] = tvals_temp.values
                        coeffs_pvals[unit_id, t, :] = pvals_temp.values
                    except Exception as e:
                        self.log.append(f'Error in fitting GLM for {unit_name} at time {time}, {e}.')
                        print(f'Error in fitting GLM for {self.session}, {unit_name} at time {time}, {e}.')
                    
                if self.model == 'glm_cpd':
                    # Measure 4: run D2 test to see value against feedback
                    try:
                        CPDs_temp, CPDs_pvals_temp = self.permutation_glm_CPD(y, predictors, self.cpd_targets)  # run permutation test

                        CPDs[unit_id, t, :] = CPDs_temp 
                        CPDs_pvals[unit_id, t, :] = CPDs_pvals_temp
                    except Exception as e:
                        self.log.append(f'Error in fitting CPD for {unit_name} at time {time}, {e}.')
                        print(f'Error in fitting CPD for {self.session}, {unit_name} at time {time}, {e}.')
                        
        # create results xarray
        self.results = self._create_results_xr(unit_ids=neural_dataset.unit.values,
                                               time_vector=time_vector,
                                               scoring=scoring)
        
        # add coeffs one by one
        # TODO this should be more that in case of a glm it adds "coeffs_feedback" and "coeff_p_vals_feedback" for each predictor, and in case of CPD it should add "CPD_feedback" and "CPD_p_vals_feedback" for each predictor. Also maybe we want both at the same time?
        predictor_names_glm = ['intercept'] + self.glm_predictors
        for i, predictor_name in enumerate(predictor_names_glm):
            self.results[f'coeffs_{predictor_name}'] = (['unit', 'time'], coeffs[:, :, i])
            self.results[f't_vals_{predictor_name}'] = (['unit', 'time'], tvals[:, :, i])
            self.results[f'p_vals_{predictor_name}'] = (['unit', 'time'], coeffs_pvals[:, :, i])

        if self.model == 'glm_cpd':
            predictor_names_cpd = self.cpd_targets
            for i, predictor_name in enumerate(predictor_names_cpd):  # TODO this can be too confusing maybe?
                if isinstance(predictor_name, list):  # if there are multiple predictors in the CPD test
                    predictor_name = '_'.join(predictor_name)
                self.results[f'CPDs_{predictor_name}'] = (['unit', 'time'], CPDs[:, :, i])
                self.results[f'CPDs_p_vals_{predictor_name}'] = (['unit', 'time'], CPDs_pvals[:, :, i])

        return self.results, self.log


