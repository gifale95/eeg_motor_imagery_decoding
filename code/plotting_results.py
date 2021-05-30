"""Plotting the motor imaging decoding accuracies onto tables.

Parameters
----------
dataset : str
		Used dataset.
sfreq : int
		Downsampling frequency of EEG data.
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
project_dir : str
		Directory of the project folder.

Output
-------
Plotting and saving the motor imaging decoding accuracies.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='5f',
		choices=['halt', '5f'])
parser.add_argument('--sfreq', default=100, type=int)
parser.add_argument('--model', type=str, default='ShallowFBCSPNet')
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/'
		'studies/dnn_bci', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> CNN BCI decoding plotting <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Loading the decoding accuracies
# =============================================================================
intra_val_results = []
intra_test_results = []
inter_val_results = []
inter_test_results = []

results_dir = os.path.join(args.project_dir, 'results', 'dataset-'+args.dataset)
subs_dir = os.listdir(results_dir)
subs_dir.sort()
for s in subs_dir:
	data_dir = os.path.join(results_dir, s, 'model-'+args.model, 'hz-'+
			format(args.sfreq,'04'))
	intra_val_results.append(np.load(os.path.join(data_dir, 'intersub-False_'+
			'data-validation_epochs-'+format(args.n_epochs,'03')+'_tbs-'+\
			format(args.batch_size,'03')+'_lr-'+format(args.lr,'05')+'_wd-'+\
			format(args.wd,'03')+'.npy'), allow_pickle=True).item()['history'])
	intra_test_results.append(np.load(os.path.join(data_dir, 'intersub-False_'+
			'data-test_epochs-'+format(args.n_epochs,'03')+'_tbs-'+\
			format(args.batch_size,'03')+'_lr-'+format(args.lr,'05')+'_wd-'+\
			format(args.wd,'03')+'.npy'), allow_pickle=True).item()['history'])
	inter_val_results.append(np.load(os.path.join(data_dir, 'intersub-True_'+
			'data-validation_epochs-'+format(args.n_epochs,'03')+'_tbs-'+\
			format(args.batch_size,'03')+'_lr-'+format(args.lr,'05')+'_wd-'+\
			format(args.wd,'03')+'.npy'), allow_pickle=True).item()['history'])
	inter_test_results.append(np.load(os.path.join(data_dir, 'intersub-True_'+
			'data-test_epochs-'+format(args.n_epochs,'03')+'_tbs-'+\
			format(args.batch_size,'03')+'_lr-'+format(args.lr,'05')+'_wd-'+\
			format(args.wd,'03')+'.npy'), allow_pickle=True).item()['history'])


# =============================================================================
# Selecting the training epochs with best decoding accuracies
# =============================================================================
intra_val_acc = np.zeros(len(subs_dir))
intra_test_acc = np.zeros(len(subs_dir))
inter_val_acc = np.zeros(len(subs_dir))
inter_test_acc = np.zeros(len(subs_dir))

for s in range(len(subs_dir)):
	for e in range(args.n_epochs):
		if intra_val_results[s][e]['valid_accuracy'] > intra_val_acc[s]:
			intra_val_acc[s] = intra_val_results[s][e]['valid_accuracy']
		if intra_test_results[s][e]['valid_accuracy'] > intra_test_acc[s]:
			intra_test_acc[s] = intra_test_results[s][e]['valid_accuracy']
		if inter_val_results[s][e]['valid_accuracy'] > inter_val_acc[s]:
			inter_val_acc[s] = inter_val_results[s][e]['valid_accuracy']
		if inter_test_results[s][e]['valid_accuracy'] > inter_test_acc[s]:
			inter_test_acc[s] = inter_test_results[s][e]['valid_accuracy']
del intra_val_results, intra_test_results, inter_val_results, inter_test_results

# Appending the averaged results
subs_dir.append('Average')
intra_val_acc = np.append(intra_val_acc, np.mean(intra_val_acc))
intra_test_acc = np.append(intra_test_acc, np.mean(intra_test_acc))
inter_val_acc = np.append(inter_val_acc, np.mean(inter_val_acc))
inter_test_acc = np.append(inter_test_acc, np.mean(inter_test_acc))

# Putting all results into one matrix
data = np.dstack((intra_val_acc, intra_test_acc, inter_val_acc,
		inter_test_acc)).squeeze()
del intra_val_acc, intra_test_acc, inter_val_acc, inter_test_acc


# =============================================================================
# Plotting the decoding accuracies onto tables
# =============================================================================
matplotlib.rcParams['font.sans-serif'] = 'Liberation Serif'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 25

colLabels = ['Intra val', 'Intra test', 'Inter val', 'Inter test']
fig, ax = plt.subplots()
ax.set_axis_off()
table = ax.table(
	cellText = data,
	rowLabels = subs_dir,
	colLabels = colLabels,
	colLoc='center',
	rowLoc='center',
	cellLoc ='center',
	loc ='upper left'
	)

ax.set_title('Decoding accuracy, dataset '+args.dataset, fontweight ="bold")













