"""
Builds a composite session ("meta session"): for each unit, m trials are randomly sampled from each class of a label
(e.g. the last three outcomes), and the units of all sessions are concatenated.

Run: python run_build_composite_session.py <config>, with <config> one of: meta_rates_history (Fig. 4a-b), meta_rates_value_4x20 (Fig. 6).
Results are saved in RESULTS_PATH/composite_sessions/<config>.

The trials are sampled randomly, so a rebuilt composite session differs slightly from the one used in the paper.
"""

import sys
import numpy
import xarray
import pandas
import matplotlib.pyplot as plt
import matplotlib
import logging
import datetime

from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.neural_data_tools import *
from popy.config import RESULTS_PATH

# %%
##% data loading
def init_log(PARAMS):
    # mkdir
    if not os.path.exists(PARAMS['floc']):
        os.makedirs(PARAMS['floc'])
    # configure logging
    logging.basicConfig(filename=os.path.join(PARAMS['floc'], 'log.txt'),
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w')  # 'w' mode will overwrite the log file

    start_time = datetime.datetime.now()
    logging.info("PARAMS:")
    for key, value in PARAMS.items():
        logging.info(f'{key}: {value}')
    logging.info(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

def end_log():
    # start time is the first log entry
    end_time = datetime.datetime.now()
    logging.info(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

def load_data_custom(monkey, session, n_extra_trials=(-1, 0)):
    # 1. Behavior data

    # Load behavior data
    behav = load_behavior(monkey, session)
    behav = drop_time_fields(behav)
    behav = add_history_of_feedback(behav, num_trials=3, one_column=True)  # add history of feedback (last 3 outcomes, Fig. 4)
    behav = add_foraging_value(behav, digitize=True, n_classes=4)  # add value function for its decoding (Fig. 6)
    #behav = add_switch_info(behav)  # add last target
    behav = behav.dropna()
    behav['feedback'] = behav['feedback'].astype('int')  # convert feedback to int
    '''behav['target'] = behav['target'].astype('int')  # convert target to int
    behav['switch'] = behav['switch'].astype('int')  # convert switch to int'''

    # 2. Neural data

    # Load neural data
    neural_data = load_neural_data(monkey, session, hz=1000)

    # remove some units
    #neural_data = remove_low_fr_neurons(neural_data, 1, print_usr_msg=False)
    neural_data = remove_trunctuated_neurons(neural_data, delay_limit=10, mode='set_nan')

    # process neural data
    neural_data = add_firing_rates(neural_data, drop_spike_trains=True, method='gauss', std=.05)
    neural_data = downsample_time(neural_data, 100)
    neural_data = time_normalize_session(neural_data)

    # 3. build neural dataset and merge with behavior
    neural_dataset = build_trial_dataset(neural_data, mode='full_trial', n_extra_trials=n_extra_trials, monkey=monkey, session=session)
    neural_dataset = merge_behavior(neural_dataset, behav)

    '''print(f"Monkey: {monkey}, Session: {session}\n",
        f"Removed {n_units_all - n_units_kept} / {n_units_all} neurons\n")'''

    return neural_dataset

# %%
CONFIGS = {
    # Fig. 4a-b: trials aligned on the last three outcomes
    'meta_rates_history': {
        'floc': os.path.join(RESULTS_PATH, 'composite_sessions', 'meta_rates_history'),
        'label': 'history_of_feedback',
        'n_extra_trials': (-1, 0),  # previous and current trial
        'm': 20
    },
    # Fig. 6: trials aligned on the value level (4 levels) and the outcome
    'meta_rates_value_4x20': {
        'floc': os.path.join(RESULTS_PATH, 'composite_sessions', 'meta_rates_value_4x20'),
        'label': 'stay_value * feedback',
        'n_extra_trials': (0, 1),  # current and next trial
        'm': 20
    },
}
PARAMS = CONFIGS[sys.argv[1]]  # e.g. python run_build_composite_session.py meta_rates_history

floc_xr = os.path.join(PARAMS['floc'], 'meta_rates.nc')

# get all sessions info
session_metadata = load_metadata()
session_metadata = session_metadata[session_metadata['block_len_valid']]

init_log(PARAMS)

# loop over sessions, get unit data
info_df = []
empty_file = True
for s, ((monkey, session), _) in enumerate(session_metadata.groupby(['monkey', 'session'])):
    print(f"{s}/{len(session_metadata)}, Processing monkey {monkey}, session {session}")
    # create labelled dataset for session
    try:
        neural_dataset = load_data_custom(monkey, session, n_extra_trials=PARAMS['n_extra_trials'])
        # add monkey and session as coordinates along dimension unitx

        if monkey == 'po' and 'target' in PARAMS['label'].split(' * '):
            # remove trials with target == 2
            neural_dataset = neural_dataset.where(neural_dataset['target'] != 2, drop=True)

    except Exception as e:
        logging.info(f"Error loading data for monkey {monkey}, session {session}: {e}")
        continue

    session_ds = []
    for unit in neural_dataset.unit.values:
        try:
            ## 1. get unit data

            # get unit data, but preserve unit dimension
            unit_index = list(neural_dataset.unit.values).index(unit)
            unit_data = neural_dataset.isel(unit=[unit_index])  # Using a list preserves 'unit' dimension

            # remove trials with NaN values
            unit_data = unit_data.dropna(dim='trial_id', how='any')
            
            ## 2. get info: number of available trials per class for this unit

            # select m trial ids from all classes
            if len(PARAMS['label'].split(' * ')) == 1:
                class_labels = [str(x) for x in unit_data[PARAMS['label']].values]
            elif len(PARAMS['label'].split(' * ')) == 2:
                class_labels_1 = [str(x) for x in unit_data[PARAMS['label'].split(' * ')[0]].values]
                class_labels_2 = [str(x) for x in unit_data[PARAMS['label'].split(' * ')[1]].values]
                class_labels = ['{} * {}'.format(x, y) for x, y in zip(class_labels_1, class_labels_2)]

            trial_ids = unit_data['trial_id'].values

            # create one row of the info_df for this unit (merge infor with number of trials)
            # get number of trials for each label
            n_trials_per_label = pandas.Series(class_labels).value_counts()
            n_trials_per_label = n_trials_per_label.sort_index()
            if len(PARAMS['label'].split(' * ')) == 1:
                renamed_counts = {f"{PARAMS['label']}_{k}": v for k, v in n_trials_per_label.to_dict().items()}
            elif len(PARAMS['label'].split(' * ')) == 2:
                label_1 = PARAMS['label'].split(' * ')[0]
                label_2 = PARAMS['label'].split(' * ')[1]
                renamed_counts = {f"{label_1}_{k.split(' * ')[0]} * {label_2}_{k.split(' * ')[1]}": v for k, v in n_trials_per_label.to_dict().items()}
            info_df_temp = {'monkey': monkey, 'session': session, 'unit': unit}
            info_df_temp.update(renamed_counts)
            info_df.append(info_df_temp.copy())

            ## 3. process neural data

            # check if all labels have at least m trials - dont use process the unit if not
            if any(n_trials_per_label < PARAMS['m']):
                print(f"Not enough trials for label {PARAMS['label']} in session {session}, unit {unit}")
                continue

            # select m trials randomly from each class (now define trial ids to select)
            trial_ids_selected = []
            for label_temp in np.sort(np.unique(class_labels)):
                # get all trial ids for this label
                trial_ids_label_tf = [class_label == label_temp for class_label in class_labels]
                trial_ids_label = trial_ids[trial_ids_label_tf]
                # select m trials randomly
                trial_ids_selected.append(numpy.random.choice(trial_ids_label, size=PARAMS['m'], replace=False))
            trial_ids_selected = numpy.concatenate(trial_ids_selected)

            # get unit_data for selected trials
            unit_data_selected = unit_data.sel(trial_id=trial_ids_selected)

            # reset_trial_id
            new_trial_ids = np.arange(len(unit_data_selected.trial_id.values))
            unit_data_selected = unit_data_selected.assign_coords(trial_id=new_trial_ids)

            # drop all coordinates along the trial axis (except for 'label')
            coords_to_drop = [coord for coord in unit_data_selected.trial_id.coords if coord not in ['trial_id']+[label for label in PARAMS['label'].split(' * ')]]
            unit_data_selected = unit_data_selected.drop(coords_to_drop)

            session_ds.append(unit_data_selected)

        except Exception as e:
            logging.info(f"Error processing unit {unit} in monkey {monkey}, session {session}: {e}")
            continue

    # concatenate all units data
    try:
        session_ds = xarray.concat(session_ds, dim='unit')
    except Exception as e:
        logging.info(f"Error concatenating data for session {session} - maybe no units left: {e}")
        continue

    # save session data to file
    try:
        if empty_file:
            # save the first unit data
            session_ds.to_netcdf(floc_xr)
            empty_file = False
        else:
            # append the unit data
            with xr.open_dataset(floc_xr) as meta_ds:
                combined_ds = xr.concat([meta_ds, session_ds], dim='unit')

            combined_ds.to_netcdf(floc_xr)

        logging.info(f"Saved data for monkey {monkey}, session {session} to {floc_xr}")
    except Exception as e:
        logging.info(f"Error saving data for monkey {monkey}, session {session}: {e}")
        continue

# save info dataframe to file
info_df = pandas.DataFrame(info_df)
info_df.to_csv(os.path.join(PARAMS['floc'], 'info_df.csv'), index=False)
end_log()
