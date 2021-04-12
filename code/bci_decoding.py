# =============================================================================
# TO DO
# =============================================================================
# 1. Error while using cropped trials with Deep4Net (bci_iv_2a) and with
	# Deep4Net or ShallowFBCSPNet (halt & 5f).
	# Change the "cropped_input_window_seconds" and "final_conv_length" params.
# 2. Reduce epoch time of HaLT and 5F datasets to 1s (best epoching window
	# seems to be [0ms 850ms]).

# 3. Dataset from (Jeong et al., 2020): gigadb.org/dataset/100788
# 4. Model hyperparameter optimization (learning rate, weight decay,
	# final_conv_length for cropped trials).
# 5. EEG hyperparameter optimization (downsampling frequency, number of used
	# channels, low- and high-frequency cuts, epoch size).
# 6. Why HFREQ data of F5 not working? Find out to have 8 subjects instead of 4.
# 7. Use other deep learning models.



"""Decoding of motor imagery states using convolutional neural networks.

Parameters
----------
dataset : str
		Used dataset ['bci_iv_2a', 'halt', '5f'].
test_sub : int
		Used test subject.
inter_subject : bool
		Whether to apply or not inter-subject learning.
cropped : bool
		Whether to use cropped trials or not.
model : str
		Used neural network model ['ShallowFBCSPNet', 'Deep4Net'].
n_epochs : int
		Number of training epochs.
lr : float
		Learning rate ['0.0625 * 0.01', '0.0001'].
wd : float
		Weight decay ['0.5 * 0.001', '0'].
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
from bci_decoding_utils import load_5f_halt
from bci_decoding_utils import windowing_data

from braindecode.util import set_random_seeds
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss

import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split


# =============================================================================
# Input parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='5f')
parser.add_argument('--test_sub', type=int, default=1)
parser.add_argument('--inter_subject', type=bool, default=True)
parser.add_argument('--cropped', type=bool, default=True)
parser.add_argument('--model', type=str, default='Deep4Net')
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=0.5 * 0.001)
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
	args.trial_start_offset_seconds = -0.5
	cropped_input_window_seconds = 4
elif args.dataset == '5f':
	args.tot_sub = 4
	args.trial_start_offset_seconds = -0.5
	cropped_input_window_seconds = 2 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
elif args.dataset == 'halt':
	args.tot_sub = 12
	args.trial_start_offset_seconds = -0.5
	cropped_input_window_seconds = 2 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# =============================================================================
# CNN training hyperparameters (as used in Schirrmeister et al., 2018)
# !!! Remove during hyperparameter optimization
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
# Loading and preprocessing the data
# =============================================================================
if args.dataset == 'bci_iv_2a':
	dataset = load_bci_iv_2a(args)
else:
	dataset = load_5f_halt(args)

# Getting EEG data info
args.n_classes = len(np.unique(dataset.datasets[0].raw.annotations.description))
args.sfreq = dataset.datasets[0].raw.info['sfreq']
args.l_freq = dataset.datasets[0].raw.info['highpass']
args.h_freq = dataset.datasets[0].raw.info['lowpass']
args.trial_start_offset_samples = int(args.trial_start_offset_seconds *
		args.sfreq)
args.nchan = dataset.datasets[0].raw.info['nchan']
args.ch_names = dataset.datasets[0].raw.info['ch_names']
if args.cropped == True:
	args.input_window_samples = int(cropped_input_window_seconds * args.sfreq)
else:
	args.input_window_samples = int(
			dataset.datasets[0].raw.annotations.duration[0]
			* args.sfreq + abs(args.trial_start_offset_samples))


# =============================================================================
# Defining the model
# =============================================================================
if args.cropped == True:
	final_conv_length = 30 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
else:
	final_conv_length = 'auto'

if args.model == 'ShallowFBCSPNet':
	model = ShallowFBCSPNet(
			in_chans=args.nchan,
			n_classes=args.n_classes,
			input_window_samples=args.input_window_samples,
			final_conv_length=final_conv_length
	)
elif args.model == 'Deep4Net':
	model = Deep4Net(
			in_chans=args.nchan,
			n_classes=args.n_classes,
			input_window_samples=args.input_window_samples,
			final_conv_length=final_conv_length
	)

# Send model to GPU
if cuda:
	model.cuda()

# Transform the model with strides to a model that outputs dense prediction, so
# it can be used to obtain predictions for all crops
if args.cropped == True:
	to_dense_prediction_model(model)


# =============================================================================
# Windowing and dividing the data into validation and training sets
# =============================================================================
# To know the models’ receptive field, we calculate the shape of model output
# for a dummy input.
if args.cropped == True:
	args.n_preds_per_input = get_output_shape(model, args.nchan,
			args.input_window_samples)[2]

valid_set, train_set = windowing_data(dataset, args)
del dataset


# =============================================================================
# Training the model
# =============================================================================
if args.cropped == True:
	clf = EEGClassifier(
		model,
		cropped=True,
		criterion=CroppedLoss,
		criterion__loss_function=torch.nn.functional.nll_loss,
		optimizer=torch.optim.AdamW,
		train_split=predefined_split(valid_set),
		optimizer__lr=args.lr,
		optimizer__weight_decay=args.wd,
		iterator_train__shuffle=True,
		batch_size=args.batch_size,
		callbacks=[
				"accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR',
				T_max=args.n_epochs - 1)),
		],
		device=args.device,
	)
else:
	clf = EEGClassifier(
			model,
			criterion=torch.nn.NLLLoss,
			optimizer=torch.optim.AdamW,
			train_split=predefined_split(valid_set),
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

