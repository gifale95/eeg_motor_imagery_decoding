"""Decoding of motor imagery states using convolutional neural networks.

Parameters
----------
dataset : str
		Used dataset.
test_sub : str
		Used test subject.
test_set : str
		Used data for testing.
inter_subject : bool
		Whether to apply or not inter-subject learning.
trial_start_offset_seconds : float
		Start offset from original trial onsets, in seconds.
model : str
		Used neural network model.
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

from bci_decoding_utils import load_5f_halt
from bci_decoding_utils import windowing_data

from braindecode.util import set_random_seeds
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss

import torch
from skorch.helper import predefined_split


# =============================================================================
# Input parameters
# =============================================================================
# 5F dataset subjects: [A, B, C, E, F, G, I]
# HaLT dataset subjects: [A, B, C, E, F, G, I, J, K, L, M]
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='5f',
		choices=['halt', '5f'])
parser.add_argument('--test_sub', type=str, default='A')
parser.add_argument('--test_set', type=str, default='validation',
		choices=['validation', 'test'])
parser.add_argument('--inter_subject', type=bool, default=True)
parser.add_argument('--trial_start_offset_seconds', type=float, default=-0.25)
parser.add_argument('--model', type=str, default='ShallowFBCSPNet')
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=128)
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

# When using cropped trials, the window size is kept at the total trial length
# (without start offset) for computational efficiency.
args.input_window_samples = int(
		dataset.datasets[0].raw.annotations.duration[0] * args.sfreq)


# =============================================================================
# Defining the model
# =============================================================================
# Now we create the model. To enable it to be used in cropped decoding
# efficiently, we manually set the length of the final convolution layer to
# some length that makes the receptive field of the ConvNet smaller than
# "input_window_samples" (e.g., "final_conv_length=30").
final_conv_length = 1

if args.model == 'ShallowFBCSPNet':
	model = ShallowFBCSPNet(
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
to_dense_prediction_model(model)


# =============================================================================
# Windowing and dividing the data into validation and training sets
# =============================================================================
# To know the models’ receptive field, we calculate the shape of model output
# for a dummy input. The model's receptive field size defines the crop size.
args.n_preds_per_input = get_output_shape(model, args.nchan,
		args.input_window_samples)[2]

valid_set, train_set = windowing_data(dataset, args)
del dataset


# =============================================================================
# Training the model
# =============================================================================
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
	callbacks=['accuracy'],
	device=args.device,
)

# Model training for a specified number of epochs. "y" is None as it is already
# supplied in the dataset.
clf.fit(train_set, y=None, epochs=args.n_epochs)


# =============================================================================
# Storing the results into a dictionary and saving
# =============================================================================
results = {
		'history': clf.history,
		'y_true': np.asarray(valid_set.get_metadata()['target']),
		'y_pred': clf.predict(valid_set),
		'args': args
}

save_dir = os.path.join(args.project_dir, 'results', 'dataset-'+args.dataset,
		'sub-'+args.test_sub, 'model-'+args.model, 'hz-'+
		format(int(args.sfreq),'04'), 'lfreq-'+str(args.l_freq)+'_hfreq-'+
		str(args.h_freq))
file_name_data = 'intersub-'+str(args.inter_subject)+'_data-'+args.test_set+\
		'_epochs-'+format(args.n_epochs,'03')+'_tbs-'+\
		format(args.batch_size,'03')+'_lr-'+format(args.lr,'05')+'_wd-'+\
		format(args.wd,'03')+'.npy'

# Creating the directory if not existing
if os.path.isdir(os.path.join(args.project_dir, save_dir)) == False:
	os.makedirs(os.path.join(args.project_dir, save_dir))
np.save(os.path.join(args.project_dir, save_dir, file_name_data), results)


# =============================================================================
# Plotting and saving the training statistics
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
fig, ax1 = plt.subplots(figsize=(20, 10))
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

# Saving the figure
file_name_plot = 'intersub-'+str(args.inter_subject)+'_data-'+args.test_set+\
		'_epochs-'+format(args.n_epochs,'03')+'_tbs-'+\
		format(args.batch_size,'03')+'_lr-'+format(args.lr,'05')+'_wd-'+\
		format(args.wd,'03')+'.jpg'
plt.savefig(os.path.join(save_dir, file_name_plot))
