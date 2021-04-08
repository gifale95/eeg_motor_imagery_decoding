def load_bci_iv_2a(args):
	"""Loading and preprocessing the validation/traning data of the BCI
	Competition IV dataset 2a dataset.

	Parameters
	----------
	args : Namespace
			Input arguments.

	Returns
	----------
	dataset : BaseConcatDataset
			BaseConcatDataset of raw MNE arrays.

	"""

	import numpy as np
	from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc
	from braindecode.datautil.preprocess import exponential_moving_standardize
	from braindecode.datasets.moabb import MOABBDataset
	from braindecode.datautil.preprocess import preprocess

	### Loading the data ###
	if args.inter_subject == False:
		dataset = MOABBDataset(dataset_name="BNCI2014001",
				subject_ids=args.test_sub)
	else:
		dataset = MOABBDataset(dataset_name="BNCI2014001",
				subject_ids=list(np.arange(1, args.tot_sub+1)))

	### Preprocessing the data ###
	# Defining the preprocessor
	preprocessors = [
			# Keep only EEG sensors
			MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False),
			# Convert from volt to microvolt, directly modifying the numpy array
			NumpyPreproc(fn=lambda x: x * 1e6),
			# Exponential moving standardization
			NumpyPreproc(fn=exponential_moving_standardize, factor_new=0.001,
					init_block_size=1000, eps=0.0001)
	]
	# Preprocessing
	preprocess(dataset, preprocessors)

	### Output ###
	return dataset



def load_5f_halt(args):
	"""Loading and preprocessing the validation/traning data of the 5F or HaLT
	datasets.

	Parameters
	----------
	args : Namespace
			Input arguments.

	Returns
	----------
	dataset : BaseConcatDataset
			BaseConcatDataset of raw MNE arrays.

	"""

	import os
	from scipy import io
	import numpy as np
	import mne
	from braindecode.datautil import exponential_moving_standardize
	from braindecode.datasets import BaseDataset, BaseConcatDataset

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

	### Subjects ###
	dataset = []
	if args.dataset == '5f':
		data_dir = os.path.join(args.project_dir, 'datasets', '5f', 'data',
				'used_data')
	elif args.dataset == 'halt':
		data_dir = os.path.join(args.project_dir, 'datasets', 'halt', 'data',
				'used_data')
	files = os.listdir(data_dir)
	files.sort()
	# Loading only one subject for intra-subject analysis
	if args.inter_subject == False:
		files = [files[args.test_sub-1]]

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
		raw_train = mne.io.RawArray(data, info)
		raw_train.info['highpass'] = 0.53
		if args.dataset == '5f':
			raw_train.info['lowpass'] = 100
			raw_train.resample(250) # resampling 5F data to 250Hz
		elif args.dataset == 'halt':
			raw_train.info['lowpass'] = 70
		del data

		### Get events ###
		events = mne.find_events(raw_train, stim_channel='stim', output='onset',
				consecutive='increasing')
		# Drop unused events
		idx = np.ones(events.shape[0], dtype=bool)
		for e in range(len(idx)):
			if events[e,2] > 6:
				idx[e] = False
		events = events[idx]
		# Drop stimuli channel
		raw_train.pick_types(eeg=True)

		### Dividing events into training and validation ###
		# The training data has 150 trials per condition, and the validation
		# data has 50 trials per condition.
		idx_train = np.zeros((events.shape[0],len(np.unique(events[:,2]))),
				dtype=bool)
		idx_val = np.zeros((events.shape[0],len(np.unique(events[:,2]))),
				dtype=bool)
		for e in range(len(np.unique(events[:,2]))):
			idx_train[np.where(events[:,2] == e+1)[0][0:100],e] = True
			idx_val[np.where(events[:,2] == e+1)[0][100:150],e] = True
		idx_train = np.sum(idx_train, 1, dtype=bool)
		idx_val = np.sum(idx_val, 1, dtype=bool)
		events_train = events[idx_train,:]
		events_val = events[idx_val,:]

		### Creating the raw data annotations ###
		if args.dataset == '5f':
			event_desc = {1: 'thumb', 2: 'index_finger', 3: 'middle_finger',
					4: 'ring_finger', 5: 'pinkie_finger'}
		elif args.dataset == 'halt':
			event_desc = {1: 'left_hand', 2: 'right_hand', 3: 'passive_neutral',
					4: 'left_leg', 5: 'tongue', 6: 'right_leg'}
		annotations_train = mne.annotations_from_events(events_train, sfreq,
				event_desc=event_desc)
		annotations_val = mne.annotations_from_events(events_val, sfreq,
				event_desc=event_desc)
		# Creating 2s trials
		annotations_train.duration = np.repeat(2., len(events_train))
		annotations_val.duration = np.repeat(2., len(events_val))
		# Adding annotations to raw data
		raw_val = raw_train.copy()
		raw_train.set_annotations(annotations_train)
		raw_val.set_annotations(annotations_val)

		### Converting to BaseConcatDataset format ###
		if args.inter_subject == False:
			i = args.test_sub-1
		description_train = {"subject": i+1, "session": 'session_T'}
		description_val = {"subject": i+1, "session": 'session_E'}
		dataset.append(BaseDataset(raw_train, description_train))
		dataset.append(BaseDataset(raw_val, description_val))
	dataset = BaseConcatDataset(dataset)

	### Output ###
	return dataset



def windowing_data(dataset, args):
	"""Windowing the preprocessed data, and dividing it into training and
	validation partitions.

	Parameters
	----------
	dataset : BaseConcatDataset
			BaseConcatDataset of raw MNE arrays.
	args : Namespace
			Input arguments.

	Returns
	----------
	valid_data : BaseConcatDataset
			BaseConcatDataset of windowed validation data.
	train_data : BaseConcatDataset
			BaseConcatDataset of windowed training data.

	"""

	from braindecode.datautil.windowers import create_windows_from_events
	from braindecode.datasets import BaseConcatDataset

	### Windowing the data ###
	# Extract sampling frequency, check that they are same in all datasets
	sfreq = dataset.datasets[0].raw.info['sfreq']
	assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
	# Calculate the trial start offset in samples.
	trial_start_offset_samples = int(args.trial_start_offset_seconds * sfreq)
	# Create windows using braindecode functions.
	windows_dataset = create_windows_from_events(
		dataset,
		trial_start_offset_samples=trial_start_offset_samples,
		trial_stop_offset_samples=0,
		preload=True,
	)
	del dataset

	### Dividing training and validation data ###
	windows_dataset = windows_dataset.split('session')
	valid_set = windows_dataset['session_E']
	train_set = windows_dataset['session_T']
	del windows_dataset

	### Selecting the right train/validation data for inter-subject analysis ###
	if args.inter_subject == True:
		valid_set = valid_set.split('subject')
		valid_set = valid_set[str(args.test_sub)]
		train_list = []
		train_set = train_set.split('subject')
		for s in range(args.tot_sub):
			if s+1 != args.test_sub:
				train_list.append(train_set[str(s+1)])
		train_set = BaseConcatDataset(train_list)

	### Output ###
	return valid_set, train_set

