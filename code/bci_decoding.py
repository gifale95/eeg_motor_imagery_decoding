# =============================================================================
# TO DO
# =============================================================================
# 1. Datasets: 5F, HaLT.
# 2. Use cropped trials.
# 3. When training the inter-subject model, make sure that each batch of
	# training data has an equal amount of trials from the different subjects.

# 4. Model hyperparameter optimization (learning rate, weight decay).
# 5. EEG hyperparameter optimization (downsampling frequency, number of used
	# channels, low- and high-frequency cuts).
# 6. Use other models.



"""Decoding of motor imagery states using convolutional neural networks.

Parameters
----------
dataset : str
		Used dataset ['bci_iv_2a', 'HaLT', '5F'].
test_sub : int
		Used test subject.
inter_subject : bool
		Whether to apply or not inter-subject generalization.
model : str
		Used neural network model ['ShallowFBCSPNet', 'Deep4Net'].
cropped : bool
		Whether to use cropped trials or not.
n_epochs : int
		Number of training epochs.
lr : float
		Learning rate ['0.0625 * 0.01', '0.0001'].
wd : float
		Weight decay ['0.5 * 0.001', '0'].
trial_start_offset_seconds : float
		Trial start offset in seconds.
batch_size : int
		Batch size for weight update.
seed : int
		Random seed to make results reproducible.
project_dir : str
		Directory of the project folder.

Output
-------
Saving of training history and decoding accuracies.

"""

import argparse
import numpy as np
import os

from bci_decoding_utils import load_bci_iv_2a
from bci_decoding_utils import load_5f
from bci_decoding_utils import load_halt

from braindecode.util import set_random_seeds
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode import EEGClassifier

import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split



# =============================================================================
# Input parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bci_iv_2a')
parser.add_argument('--test_sub', type=int, default=1)
parser.add_argument('--inter_subject', type=bool, default=False)
parser.add_argument('--model', type=str, default='ShallowFBCSPNet')
parser.add_argument('--cropped', type=bool, default=False)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0625 * 0.01)
parser.add_argument('--wd', type=float, default=0.5 * 0.001)
parser.add_argument('--trial_start_offset_seconds', type=float, default=-0.5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=20200220)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/'
		'studies/dnn_bci', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> CNN BCI decoding <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))



# =============================================================================
# Dataset-specific parameters
# =============================================================================
if args.dataset == 'bci_iv_2a':
	args.tot_sub = 9
elif args.dataset == '5f':
	args.tot_sub = 8
elif args.dataset == 'halt':
	args.tot_sub = 12



# =============================================================================
# !!! Model-specific parameters (remove during hyperparameter optimization)
# =============================================================================
# These values were found to be good for shallow and deep network.
if args.model == 'ShallowFBCSPNet':
	args.lr = 0.0625 * 0.01
	args.wd = 0
elif args.model == 'Deep4Net':
	args.lr = 1 * 0.01
	args.wd = 0.5 * 0.001



# =============================================================================
# GPU/CPU and random seed
# =============================================================================
# Check for GPU and set random seed to make results reproducible
cuda = torch.cuda.is_available()
args.device = 'cuda' if cuda else 'cpu'
if cuda:
	torch.backends.cudnn.benchmark = True
set_random_seeds(seed=args.seed, cuda=cuda)



# =============================================================================
# Loading, preprocessing and windowing the data
# =============================================================================
if args.dataset == 'bci_iv_2a':
	valid_set, train_set = load_bci_iv_2a(args)
elif args.dataset == '5f':
	valid_set, train_set = load_5f(args)
elif args.dataset == 'halt':
	valid_set, train_set = load_halt(args)


# Getting EEG data info
args.freq = valid_set.datasets[0].windows.info['sfreq']
args.l_freq = valid_set.datasets[0].windows.info['highpass']
args.h_freq = valid_set.datasets[0].windows.info['lowpass']
args.trial_start_offset_samples = int(args.trial_start_offset_seconds *
		args.freq)
args.in_chans = valid_set.datasets[0].windows.info['nchan']
args.ch_names = valid_set.datasets[0].windows.info['ch_names']
args.input_window_samples = valid_set[0][0].shape[1]
args.n_classes = len(np.unique(valid_set.get_metadata()['target']))



# =============================================================================
# Construct model
# =============================================================================
# Defining the model
if args.model == 'ShallowFBCSPNet':
	model = ShallowFBCSPNet(
			in_chans=args.in_chans,
			n_classes=args.n_classes,
			input_window_samples=args.input_window_samples,
			final_conv_length='auto'
	)
elif args.model == 'Deep4Net':
	model = Deep4Net(
			in_chans=args.in_chans,
			n_classes=args.n_classes,
			input_window_samples=args.input_window_samples,
			final_conv_length='auto'
	)



# =============================================================================
# Training the model
# =============================================================================
clf = EEGClassifier(
	model,
	criterion=torch.nn.NLLLoss,
	optimizer=torch.optim.AdamW,
	train_split=predefined_split(valid_set), # using valid_set for validation
	optimizer__lr=args.lr,
	optimizer__weight_decay=args.wd,
	batch_size=args.batch_size,
	callbacks=[
			"accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR',
			T_max=args.n_epochs - 1)),
	],
	device=args.device,
)

# Model training for a specified number of epochs. "y" is None as it is already
# supplied in the dataset.
clf.fit(train_set, y=None, epochs=args.n_epochs)



# =============================================================================
# Storing the results into a dictionary
# =============================================================================
results = {
		"history": clf.history,
		"y_true": np.asarray(valid_set.get_metadata()["target"]),
		"y_pred": clf.predict(valid_set),
		"args": args
}



# =============================================================================
# Saving the results
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'dataset-'+
		args.dataset, 'sub-'+format(args.test_sub, '02'), 'inter_subject-'+
		str(args.inter_subject), 'model-'+args.model, 'cropped-'+
		str(args.cropped), 'hz-'+format(int(args.sfreq), '04'), 'lfreq-'+
		str(args.l_freq)+'_hfreq-'+str(args.h_freq))
file_name = 'epochs-'+format(args.n_epochs, '03')+'_lr-'+str(args.lr)+'_wd-'+\
		str(args.wd)+'.npy'

# Creating the directory if not existing
# if os.path.isdir(os.path.join(args.project_dir, save_dir)) == False:
# 	os.makedirs(os.path.join(args.project_dir, save_dir))
# np.save(os.path.join(args.project_dir, save_dir, file_name), results)

