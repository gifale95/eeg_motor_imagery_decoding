# =============================================================================
# TO DO
# =============================================================================
# 1. Change Deep4Net parameters so that it works with 5f/halt data.

# 2. Training:
	# - Gridsearch over weight decay, learning rate, batch size.
		# Learning rate: [0.0001 10, 0.05]
		# Weight decay: [0.0001 0.1]
		# Batch size: [16, 32, 64, 128]
	# - Save (only) the learning plot for each model training, with parameters
		# on file name.

# 3. Model hyperparameter optimization (learning rate, weight decay, batch
	# size, kernel sizes, Adam's parameters, dropout).
# 4. EEG hyperparameter optimization (window_size, downsampling frequency,
	# number of used channels, low- and high-frequency cuts).
# 5. Why HFREQ data of F5 not working? Fix to have 8 subjects instead of 4.
# 6. Dataset from (Jeong et al., 2020): gigadb.org/dataset/100788
# 7. Data augmentation techniques (also beneficial for regularization).
# 8. Use other deep learning models.



"""Decoding of motor imagery states using convolutional neural networks.

Parameters
----------
dataset : str
		Used dataset ['bci_iv_2a', 'halt', '5f'].
test_sub : int
		Used test subject.
test_set : str
		Used data for testing ['validation', 'test'].
inter_subject : bool
		Whether to apply or not inter-subject learning.
cropped : bool
		Whether to use cropped trials or not.
model : str
		Used neural network model ['ShallowFBCSPNet', 'Deep4Net'].
n_epochs : int
		Number of training epochs.
lr : float
		Learning rate.
wd : float
		Weight decay coefficient.
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
parser.add_argument('--dataset', type=str, default='bci_iv_2a')
parser.add_argument('--test_sub', type=int, default=1)
parser.add_argument('--test_set', type=str, default='validation')
parser.add_argument('--inter_subject', type=bool, default=True)
parser.add_argument('--cropped', type=bool, default=False)
parser.add_argument('--model', type=str, default='ShallowFBCSPNet')
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
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
elif args.dataset == '5f':
	args.tot_sub = 4
	args.trial_start_offset_seconds = -0.25
elif args.dataset == 'halt':
	args.tot_sub = 12
	args.trial_start_offset_seconds = -0.25


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
# For intra-subject decoding 4/6 of the data of the subject of interest is used
# for training, 1/6 for validation and 1/6 for testing.
# For inter-subject decoding 1/2 of the data of the subject of interest is used
# for validation and 1/2 for testing. All the data from the other subjects is
# used for training.
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

# When using full trials, the window size is given by the total trial length
# plus its start offset.
# When using cropped trials, the window size is kept at the total trial length
# (without start offset) for computational efficiency.
if args.cropped == True:
	args.input_window_samples = int(
			dataset.datasets[0].raw.annotations.duration[0] * args.sfreq)
else:
	args.input_window_samples = int(
			dataset.datasets[0].raw.annotations.duration[0]
			* args.sfreq + abs(args.trial_start_offset_samples))


# =============================================================================
# Defining the model
# =============================================================================
# Now we create the model. To enable it to be used in cropped decoding
# efficiently, we manually set the length of the final convolution layer to
# some length that makes the receptive field of the ConvNet smaller than
# "input_window_samples" (e.g., "final_conv_length=30").
if args.cropped == False:
	final_conv_length = 'auto'
else:
	final_conv_length = 1

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
# for a dummy input. The model's receptive field size defines the crop size.
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
#results = {
		#"history": clf.history,
		#"y_true": np.asarray(valid_set.get_metadata()["target"]),
		#"y_pred": clf.predict(valid_set),
		#"args": args
#}


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


# =============================================================================
# Plotting the training statistics
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'train_accuracy',
		'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
		index=clf.history[:, 'epoch'])

# Get percent of misclass for better visual comparison to loss
df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
		valid_misclass=100 - 100 * df.valid_accuracy)

plt.style.use('seaborn')
fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
		ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False,
		fontsize=14)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['train_misclass', 'valid_misclass']].plot(
		ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
ax2.set_ylim(ax2.get_ylim()[0], 85) # make some room for legend
ax1.set_xlabel("Epoch", fontsize=14)

# Where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-',
		label='Train'))
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':',
		label='Valid'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()
