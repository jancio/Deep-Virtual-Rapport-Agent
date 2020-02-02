#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Helper functions for rapport model training/testing.
#######################################################################################################################


from datetime import timedelta
import time
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score


def time_diff(start_time):
	"""Get time difference between the current time and start time.

	Args:
		start_time (float): Start time in seconds since epoch

	Returns:
		(string): Time difference in seconds between the current and start time
	"""
	return str(timedelta(seconds=time.time() - start_time))


def calculate_metrics(predictions, labels, labels_names, losses=None, loss_function_pos_weights=None):
	"""Calculate all desired evaluation metrics on predictions aggregated from all batches.

	Currently, the evaluation metrics are: balanced accuracy, F1 score, precision, and recall.
	When called during training or validation, the losses argument is expected.
	The losses argument is not provided during testing.

	Args:
		predictions (array/list): 2D (number of samples x number of labels/targets) array of predictions
		labels (array/list): 2D (number of samples x number of labels/targets) array of ground-truth labels
		labels_names (array/list): Names of labels/targets
		losses (array/list): 2D (number of samples x number of labels/targets) array of losses, already weighted for
			each class independently
		loss_function_pos_weights (array): Positive class weights (one weight for each class)

	Returns:
		metrics (dict of dict of lists): Dictionary that maps class/label/target name and metric name to a
			corresponding metric value. Besides the class/label/target names provided in labels_names, it also contains
			a key 'overall' referring to the average over all targets/labels.
	"""

	# Convert inputs to numpy arrays if they are lists
	if type(predictions) == list: 
		predictions = np.array(predictions)
	if type(labels) == list: 
		labels = np.array(labels)
	if type(losses) in [list, np.ndarray]: 
		# Reduce/average loss for each class
		# In case of reduction='none', the loss needs to be normalized manually
		pos_labels_cnt = np.sum(labels, axis=0)
		neg_labels_cnt = len(labels) - pos_labels_cnt
		class_losses = np.sum(losses, axis=0) / (loss_function_pos_weights * pos_labels_cnt + 1. * neg_labels_cnt)

	metrics = dict()

	for label_idx, label_name in enumerate(labels_names):
		metrics[label_name] = dict(
			bacc      = balanced_accuracy_score(labels[:, label_idx], predictions[:, label_idx]), 
			f1        = f1_score(labels[:, label_idx], predictions[:, label_idx], average='binary'), 
			precision = precision_score(labels[:, label_idx], predictions[:, label_idx]),
			recall    = recall_score(labels[:, label_idx], predictions[:, label_idx])
		)
		if losses is not None:
			metrics[label_name]['loss'] = class_losses[label_idx]

	# Add overall metrics, taking into consideration all labels
	metrics['overall'] = dict()
	for metric_name in metrics[labels_names[0]].keys():
		metrics['overall'][metric_name] = np.mean([metrics[label_name][metric_name] for label_name in labels_names])

	return metrics
