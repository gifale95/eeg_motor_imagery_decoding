# =============================================================================
# TO DO
# =============================================================================
# 1. Divide the data into training (2/3 of data) and validation (1/3 of data)
	# partitions.
# 2. Load all subjects for inter-subject anaylsis.



"""Loading, preprocessing and windowing the validation/traning data of the HaLT
dataset.

Parameters
----------
args : Namespace
		Input arguments.

Returns
----------
valid_set : BaseConcatDataset
		Validation data.
train_set : BaseConcatDataset
		Training data.

"""

import argparse
import os
from scipy import io
import numpy as np
import mne
from braindecode.datautil import exponential_moving_standardize
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.datautil.windowers import create_windows_from_events


### Input parameters ###
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/'
		'studies/dnn_bci', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> Preprocessing the HaLT data <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


### Subjects and channel types ###
data_dir = os.path.join(args.project_dir, 'datasets', 'halt', 'data',
		'used_data')
files = os.listdir(data_dir)
files.sort()

# Rejecting channels A1, A1, X5 (see paper)
ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7",
		"F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "stim"]
ch_types = ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
		"eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
		"stim"]
idx_chan = np.ones(22, dtype=bool)
unused_chans = np.asarray((10, 11, 21))
idx_chan[unused_chans] = False


### Loading and preprocessing the .mat data ###
for i, file in enumerate(files):
	data = io.loadmat(os.path.join(data_dir, file), chars_as_strings=True)['o']
	sfreq = np.asarray(data[0][0]['sampFreq'][0])
	marker = np.transpose(np.asarray(data[0][0]['marker']))
	data = np.transpose(np.asarray(data[0][0]['data']))[idx_chan,:]
	data = exponential_moving_standardize(data)
	data = np.append(data, marker, 0)
	del marker


### Converting to MNE format ###
	info = mne.create_info(ch_names, sfreq, ch_types)
	raw = mne.io.RawArray(data, info)
	raw.info['highpass'] = 0.53
	raw.info['lowpass'] = 70
	del data


### Creating the raw data annotations ###
	# Get events and drop stimuli channel
	events = mne.find_events(raw, stim_channel='stim', output='onset',
			consecutive='increasing')
	idx = np.ones(events.shape[0], dtype=bool)
	for e in range(len(idx)):
		if events[e,2] > 6:
			idx[e] = False
	events = events[idx]
	raw.pick_types(eeg=True)

	# Make annotations
	event_desc = {1: 'left_hand', 2: 'right_hand', 3: 'passive_neutral',
			4: 'left_leg', 5: 'tongue', 6: 'right_leg'}
	annotations = mne.annotations_from_events(events, sfreq,
			event_desc=event_desc)
	annotations.duration = np.repeat(2.5, len(events)) # 2 seconds trials
	raw.set_annotations(annotations)


### Converting to BaseConcatDataset format ###
	description = {"subject": i+1}
	dataset = BaseDataset(raw, description)
	dataset = BaseConcatDataset([dataset])
	del raw


### Windowing the data ###
	# Windowing arguments
	trial_start_offset_seconds = -0.5
	trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

	# Create WindowsDatasets from mne.RawArrays
	windows_dataset = create_windows_from_events(
		dataset,
		trial_start_offset_samples=trial_start_offset_samples,
		trial_stop_offset_samples=0,
		preload=True,
	)

