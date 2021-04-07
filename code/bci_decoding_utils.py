# =============================================================================
# Defining the preprocessing steps for the different datasets
# =============================================================================
def prepr_definition(args):
	"""Defining the preprocessing steps for the different datasets.

	Parameters
	----------
	args : Namespace
			Input arguments.

	Returns
	-------
	Preprocessor.

	"""

	from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc
	from braindecode.datautil.preprocess import exponential_moving_standardize


### Defining the preprocessor ###
	if args.dataset == 'bci_iv_2a':
		preprocessors = [
				# Keep only EEG sensors
				MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False),
				# Convert from volt to microvolt, directly modifying the numpy array
				NumpyPreproc(fn=lambda x: x * 1e6),
				# Bandpass filter
				MNEPreproc(fn='filter', l_freq=args.l_freq, h_freq=args.h_freq),
				# Exponential moving standardization
				NumpyPreproc(fn=exponential_moving_standardize, factor_new=0.001,
						init_block_size=1000, eps=0.0001)
		]

### Output ###
	return preprocessors



# =============================================================================
# Loading, preprocessing and windowing the validation/training data
# =============================================================================
def load_data(args, preprocessors):
	"""Loading, preprocessing and windowing the validation/traning data.

	Parameters
	----------
	args : Namespace
			Input arguments.
	preprocessors : list
			Preprocessors to apply to the EEG data.

	Returns
	-------
	Preprocessed and windowed validation and training data.

	"""

	import numpy as np
	from braindecode.datasets.moabb import MOABBDataset
	from braindecode.datautil.preprocess import preprocess
	from braindecode.datautil.windowers import create_windows_from_events

	print('\n\n>>> Loading, preprocessing and windowing the data <<<')


### Validation data ###
	# Loading the data
	if args.dataset == 'bci_iv_2a':
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
	if args.dataset == 'bci_iv_2a':
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
		if args.dataset == 'bci_iv_2a':
			train_set = train_set.split('session')
			train_set = train_set['session_T']
	else:
		# Extracting the training data
		if args.dataset == 'bci_iv_2a':
			train_set = sub_set['session_T']


### Output ###
	return valid_set, train_set

