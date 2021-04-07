def load_bci_iv_2a(args):
	"""Loading, preprocessing and windowing the validation/traning data of the
	BCI Competition IV dataset 2a dataset.

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

	import numpy as np
	from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc
	from braindecode.datautil.preprocess import exponential_moving_standardize
	from braindecode.datasets.moabb import MOABBDataset
	from braindecode.datautil.preprocess import preprocess
	from braindecode.datautil.windowers import create_windows_from_events


### Defining the preprocessor ###
	preprocessors = [
			# Keep only EEG sensors
			MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False),
			# Convert from volt to microvolt, directly modifying the numpy array
			NumpyPreproc(fn=lambda x: x * 1e6),
			# Exponential moving standardization
			NumpyPreproc(fn=exponential_moving_standardize, factor_new=0.001,
					init_block_size=1000, eps=0.0001)
	]


### Validation data ###
	# Loading the data
	sub_set = MOABBDataset(dataset_name="BNCI2014001",
			subject_ids=args.test_sub)
	# Preprocessing the data
	preprocess(sub_set, preprocessors)
	# Extract sampling frequency, check that they are same in all datasets
	sfreq = sub_set.datasets[0].raw.info['sfreq']
	assert all([ds.raw.info['sfreq'] == sfreq for ds in sub_set.datasets])
	# Calculate the trial start offset in samples.
	trial_start_offset_samples = int(args.trial_start_offset_seconds * sfreq)
	# Create windows using braindecode functions.
	sub_set = create_windows_from_events(
		sub_set,
		trial_start_offset_samples=trial_start_offset_samples,
		trial_stop_offset_samples=0,
		preload=True,
	)
	# Extracting the validation data
	sub_set = sub_set.split('session')
	valid_set = sub_set['session_E']


### Training data ###
	if args.inter_subject == True:
		# Loading the data
		args.train_sub = list(np.delete(np.arange(1, args.tot_sub+1),
				args.test_sub-1))
		if args.dataset == 'bci_iv_2a':
			train_set = MOABBDataset(dataset_name="BNCI2014001",
					subject_ids=args.train_sub)
		# Preprocessing the data
		preprocess(train_set, preprocessors)
		# Check that the frequencies are same in all datasets
		assert all([ds.raw.info['sfreq'] == sfreq for ds in
				train_set.datasets])
		# Create windows using braindecode functions.
		train_set = create_windows_from_events(
			train_set,
			trial_start_offset_samples=trial_start_offset_samples,
			trial_stop_offset_samples=0,
			preload=True,
		)
		# Extracting the training data
		train_set = train_set.split('session')
		train_set = train_set['session_T']
	else:
		# Extracting the training data
		train_set = sub_set['session_T']


### Output ###
	return valid_set, train_set



def load_5f_halt(args):
# =============================================================================
# TO DO
# =============================================================================
# 1. Divide the data into training (2/3 of data) and validation (1/3 of data)
	# partitions.
# 2. Load all subjects for inter-subject anaylsis.
	"""Loading, preprocessing and windowing the validation/traning data of the
	5F or HaLT dataset.

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

	import os
	from scipy import io
	import numpy as np
	import mne
	from braindecode.datautil import exponential_moving_standardize
	from braindecode.datasets import BaseDataset, BaseConcatDataset
	from braindecode.datautil.windowers import create_windows_from_events


### Subjects ###
	if args.dataset == '5f':
		data_dir = os.path.join(args.project_dir, 'datasets', '5f', 'data',
				'used_data')
	elif args.dataset == 'halt':
		data_dir = os.path.join(args.project_dir, 'datasets', 'halt', 'data',
				'used_data')
	files = os.listdir(data_dir)
	files.sort()


### Channel types ###
# Rejecting channels A1, A1, X5 (see paper)
	ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
			"F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "stim"]
	ch_types = ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
			"eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
			"eeg", "stim"]
	idx_chan = np.ones(22, dtype=bool)
	unused_chans = np.asarray((10, 11, 21))
	idx_chan[unused_chans] = False


### Loading and preprocessing the .mat data ###
	for i, file in enumerate(files):
		data = io.loadmat(os.path.join(data_dir, file),
				chars_as_strings=True)['o']
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
		if args.dataset == '5f':
			raw.info['lowpass'] = 100
		elif args.dataset == 'halt':
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
		if args.dataset == '5f':
			event_desc = {1: 'thumb', 2: 'index_finger', 3: 'middle_finger',
					4: 'ring_finger', 5: 'pinkie_finger'}
		elif args.dataset == 'halt':
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


### Output ###
	#return valid_set, train_set
