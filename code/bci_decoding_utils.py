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
		data_dir = os.path.join(args.project_dir, 'datasets', '5f', 'data')
	elif args.dataset == 'halt':
		data_dir = os.path.join(args.project_dir, 'datasets', 'halt', 'data')
	files = os.listdir(data_dir)
	files.sort()
	# Loading only one subject for intra-subject analysis
	if args.inter_subject == False:
		files = [files[args.test_sub-1]]

	### Loading and preprocessing the .mat data ###
	for i, file in enumerate(files):
		print('\n\nData file --> '+file+'\n\n')
		data = io.loadmat(os.path.join(data_dir, file),
				chars_as_strings=True)['o']
		sfreq = np.asarray(data[0][0]['sampFreq'][0])
		marker = np.transpose(np.asarray(data[0][0]['marker']))
		data = np.transpose(np.asarray(data[0][0]['data']))[idx_chan,:]
		data = exponential_moving_standardize(data)
		data = np.append(data, marker, 0)
		del marker

		### Converting to MNE format and downsample ###
		info = mne.create_info(ch_names, sfreq, ch_types)
		raw_train = mne.io.RawArray(data, info)
		#raw_train.info['highpass'] = 0.53
		#raw_train.info['lowpass'] = 70
		del data

		### Get events and downsample data ###
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
		# Downsampling the data to 100Hz
		raw_train.resample(100)

		### Dividing events into training, validation and test ###
		# For intra-subject decoding 4/6 of the data of the subject of interest
		# is used for training, 1/6 for validation and 1/6 for testing.
		# For inter-subject decoding 1/2 of the data of the subject of interest
		# is used for validation and 1/2 for testing. All the data from the
		# other subjects is used for training.
		idx_train = np.zeros((events.shape[0],len(np.unique(events[:,2]))),
				dtype=bool)
		idx_val = np.zeros((events.shape[0],len(np.unique(events[:,2]))),
				dtype=bool)
		idx_test = np.zeros((events.shape[0],len(np.unique(events[:,2]))),
				dtype=bool)
		for e in range(len(np.unique(events[:,2]))):
			if args.inter_subject == False:
				idx_train[np.where(events[:,2] == e+1)[0][0:100],e] = True
				idx_val[np.where(events[:,2] == e+1)[0][100:125],e] = True
				idx_test[np.where(events[:,2] == e+1)[0][125:150],e] = True
			else:
				if args.test_sub == i+1:
					idx_val[np.where(events[:,2] == e+1)[0][0:75],e] = True
					idx_test[np.where(events[:,2] == e+1)[0][75:150],e] = True
				else:
					idx_train[np.where(events[:,2] == e+1)[0][0:150],e] = True
		idx_train = np.sum(idx_train, 1, dtype=bool)
		idx_val = np.sum(idx_val, 1, dtype=bool)
		idx_test = np.sum(idx_test, 1, dtype=bool)
		events_train = events[idx_train,:]
		events_val = events[idx_val,:]
		events_test = events[idx_test,:]

		### Creating the raw data annotations ###
		if args.dataset == '5f':
			event_desc = {1: 'thumb', 2: 'index_finger', 3: 'middle_finger',
					4: 'ring_finger', 5: 'pinkie_finger'}
		elif args.dataset == 'halt':
			event_desc = {1: 'left_hand', 2: 'right_hand', 3: 'passive_neutral',
					4: 'left_leg', 5: 'tongue', 6: 'right_leg'}
		if args.inter_subject == False:
			annotations_train = mne.annotations_from_events(events_train, sfreq,
					event_desc=event_desc)
			annotations_val = mne.annotations_from_events(events_val, sfreq,
					event_desc=event_desc)
			annotations_test = mne.annotations_from_events(events_test, sfreq,
					event_desc=event_desc)
			# Creating 1s trials
			annotations_train.duration = np.repeat(1., len(events_train))
			annotations_val.duration = np.repeat(1., len(events_val))
			annotations_test.duration = np.repeat(1., len(events_test))
			# Adding annotations to raw data
			raw_val = raw_train.copy()
			raw_test = raw_train.copy()
			raw_train.set_annotations(annotations_train)
			raw_val.set_annotations(annotations_val)
			raw_test.set_annotations(annotations_test)
		else:
			if args.test_sub == i+1:
				annotations_val = mne.annotations_from_events(events_val, sfreq,
						event_desc=event_desc)
				annotations_test = mne.annotations_from_events(events_test,
						sfreq, event_desc=event_desc)
				# Creating 1s trials
				annotations_val.duration = np.repeat(1., len(events_val))
				annotations_test.duration = np.repeat(1., len(events_test))
				# Adding annotations to raw data
				raw_val = raw_train.copy()
				raw_test = raw_train.copy()
				raw_val.set_annotations(annotations_val)
				raw_test.set_annotations(annotations_test)
			else:
				annotations_train = mne.annotations_from_events(events_train,
						sfreq, event_desc=event_desc)
				# Creating 1s trials
				annotations_train.duration = np.repeat(1., len(events_train))
				# Adding annotations to raw data
				raw_train.set_annotations(annotations_train)

		### Converting to BaseConcatDataset format ###
		if args.inter_subject == False:
			i = args.test_sub-1
		description_train = {"subject": i+1, "session": 'training'}
		description_val = {"subject": i+1, "session": 'validation'}
		description_test = {"subject": i+1, "session": 'test'}
		if args.inter_subject == False:
			dataset.append(BaseDataset(raw_train, description_train))
			dataset.append(BaseDataset(raw_val, description_val))
			dataset.append(BaseDataset(raw_test, description_test))
		else:
			if args.test_sub == i+1:
				dataset.append(BaseDataset(raw_val, description_val))
				dataset.append(BaseDataset(raw_test, description_test))
			else:
				dataset.append(BaseDataset(raw_train, description_train))
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

	### Windowing the data ###
	# Extract sampling frequency, check that they are same in all datasets
	sfreq = dataset.datasets[0].raw.info['sfreq']
	assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
	# Create windows using braindecode functions.
	if args.cropped == True:
		windows_dataset = create_windows_from_events(
			dataset,
			trial_start_offset_samples=args.trial_start_offset_samples,
			trial_stop_offset_samples=0,
			window_size_samples=args.input_window_samples,
			window_stride_samples=args.n_preds_per_input,
			drop_last_window=False,
			preload=True,
		)
	else:
		windows_dataset = create_windows_from_events(
			dataset,
			trial_start_offset_samples=args.trial_start_offset_samples,
			trial_stop_offset_samples=0,
			preload=True,
		)
	del dataset

	### Dividing training, validation and test data ###
	windows_dataset = windows_dataset.split('session')
	train_set = windows_dataset['training']
	if args.test_set == 'validation':
		valid_set = windows_dataset['validation']
	else:
		valid_set = windows_dataset['test']

	### Output ###
	return valid_set, train_set

