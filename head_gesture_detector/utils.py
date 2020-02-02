#######################################################################################################################
# Project: Deep Virtual Rapport Agent (head gesture detector)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Helper functions for the Head Gesture Detector (HGD) model training/testing.
#
#     - Evaluate custom metrics. 
#     - Plot training and validation curves for various evaluation metrics.
#     - Plot test results by subjects.
#     - Save and load training history. 
#######################################################################################################################


from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from keras.callbacks import Callback, History
from collections import defaultdict
import numpy as np
import scipy.signal
import os
import pickle

# Shortest head nod annotations (6 of them) in vra1 dataset are 0.29 sec (0.29 sec x 30 FPS = 9 frames)
# Can also refer to ICMI paper (same setting!)
# (5 or more consecutive frames make a head gesture not to be filtered out)
SMOOTHING_KERNEL_SIZE = 9

# Voting strategies to fuse multiple predictions (for the same frame) from adjacent output sequences 
voting_strategies = ['last', 'majority']

# Evaluation metrics
metrics_names = ['bacc', 'f1', 'precision', 'recall']


class CustomMetrics(Callback):
    '''
    Custom metric added to logs on each epoch end.
        e.g. the key 'val_majority_f1' in history.hist gives the list of validation set f1 scores for each epoch when majority voting strategy was used
    '''
    def __init__(self, window_size, training_data, train_lens, val_lens):
        self.window_size = window_size
        # Training (non-shuffled) data has to be provided, unlike validation data
        self.training_data = training_data
        # Chunk lengths for training data have to be flattened, since they are provided as N x 2 array that contains the lengths of chunks before and after the validation chunk
        self.train_lens = train_lens.flatten()
        self.val_lens = val_lens
        super(CustomMetrics, self).__init__()
                            
    def on_epoch_end(self, epoch, logs={}):
        epoch_metrics = dict()
        epoch_metrics['train'] = evaluate_custom_metrics(Y_true=self.training_data[1], 
                                              Y_pred=self.model.predict_classes(self.training_data[0], batch_size=10000), 
                                              chunk_lens=self.train_lens, window_size=self.window_size)
        epoch_metrics['val'] = evaluate_custom_metrics(Y_true=self.validation_data[1], 
                                              Y_pred=self.model.predict_classes(self.validation_data[0], batch_size=len(self.validation_data[0])), 
                                              chunk_lens=self.val_lens, window_size=self.window_size)
    
        # For each dataset partition and each voting strategy, calculate all metrics
        for dataset_partition in ['train', 'val']:
            for vs in voting_strategies:
                print(f"[{vs}] ", end="")
                for mn in metrics_names:
                    logs[f'{dataset_partition}_{vs}_{mn}'] = epoch_metrics[dataset_partition][vs][mn]
                    print(f"- {dataset_partition}_{mn}: {logs[f'{dataset_partition}_{vs}_{mn}']:.4f} ", end="")
                print()


def evaluate_custom_metrics(Y_true, Y_pred, chunk_lens, window_size, smooth=False):
    
    # Ground-truths as 1D array: take segment's last label, for each frame
    y_true = Y_true[:, -1, 0].astype(int)
    
    # Full predictions are 3D array (Y_pred)
    # Apply voting strategies to Y_pred to obtain 1D predictions y_pred
    y_pred = dict()        

    #########################################################################
    # Voting strategy: LAST

    y_pred['last'] = Y_pred[:, -1, 0]

    #########################################################################
    # Voting strategy: MAJORITY

    y_pred['majority'] = np.zeros(len(y_true), dtype=int)
    assert np.sum(chunk_lens) == len(Y_pred)

    chunk_offset = 0
    for chunk_len in chunk_lens:

        # List of lists, each inner list contains votes (0 or 1)
        votes = [[] for _ in range(chunk_len + window_size - 1)]
        votes_idx = 0

        for segment in Y_pred[chunk_offset:chunk_offset + chunk_len]:
            for pred_label in segment:
                # Append vote (0 or 1)
                votes[votes_idx].append( int(pred_label[0]) )
                votes_idx += 1
            # Reset pointer for the next segment
            votes_idx = votes_idx - window_size + 1

        # Perform majority voting over votes (skipping the first window_size - 1)
        for idx, perframe_votes in enumerate(votes[window_size:]):
            neg_cnt = perframe_votes.count(0)
            pos_cnt = perframe_votes.count(1)
            # In case of tie, use label from the last-frame voting strategy
            if neg_cnt == pos_cnt:
                majority_vote = y_pred['last'][chunk_offset + idx]
            elif neg_cnt < pos_cnt:
                majority_vote = 1
            else:
                majority_vote = 0
            y_pred['majority'][chunk_offset + idx] = majority_vote

        chunk_offset += chunk_len      

    assert len(y_pred['majority']) == len(y_pred['last'])
    
    # Apply smoothing (median filter), then convert back to integers
    if smooth:
        y_pred['last'] = scipy.signal.medfilt(y_pred['last'], kernel_size=SMOOTHING_KERNEL_SIZE).astype(int)
        y_pred['majority'] = scipy.signal.medfilt(y_pred['majority'], kernel_size=SMOOTHING_KERNEL_SIZE).astype(int)

#     print(list(y_true))
#     print(list(y_pred['last']))
#     print(list(y_pred['majority'].astype(int)))
#     print(type(y_true[0]), type(y_pred['last'][0]), type(y_pred['majority'][0]))
    
    # Calculate metrics for each voting strategy
    custom_metrics_dict_dict = dict()
    for vs in voting_strategies:
        custom_metrics_dict_dict[vs] = dict(
            bacc      = balanced_accuracy_score(y_true, y_pred[vs]), 
            f1        = f1_score(y_true, y_pred[vs], average='binary'), 
            precision = precision_score(y_true, y_pred[vs]),
            recall    = recall_score(y_true, y_pred[vs])
        )
        
    return custom_metrics_dict_dict


def arch_to_str(arch):
    """Convert network architecture from list of integers to string."""
    return '-'.join(str(n_units) for n_units in arch)


def plot_test_results_by_subject(data, metric_name, n_folds, title):
    
    dataset_split_filename = f'/home/ICT2000/jondras/dvra_datasets/vra1/subject_independent/dataset_split_{n_folds}fold.npz'
    x_ticks = [np.load(dataset_split_filename)[f'{k}_test'][0].split('/')[-1].split('.')[0][3:] for k in range(n_folds)]
    x_axis = np.arange(len(x_ticks))    
    
    plt.figure(figsize=[16,6])
    w = 0.4
    plt.bar(x_axis - w/2, data['last'][metric_name], align='center', width=w, 
            label=f"last VS")
#             label=f"last VS: {np.mean(data['last'][metric_name])} +/- {np.std(data['last'][metric_name])}")
    plt.bar(x_axis + w/2, data['majority'][metric_name], align='center', width=w, 
            label=f"majority VS")
#             label=f"majority VS: {np.mean(data['majority'][metric_name])} +/- {np.std(data['majority'][metric_name])}")
    plt.axhline(y=np.mean(data['last'][metric_name]), c='blue', label=f'mean={np.mean(data["last"][metric_name]):.4f} +/- {np.std(data["last"][metric_name]):.4f}')
    plt.axhline(y=np.mean(data['majority'][metric_name]), c='orange', label=f'mean={np.mean(data["majority"][metric_name]):.4f} +/- {np.std(data["majority"][metric_name]):.4f}')
    plt.axhline(y=0.5, c='k', linestyle='--', label='baseline=0.5')
    plt.xticks(x_axis, x_ticks, rotation=90)
    plt.xlabel('Subject ID')#,fontsize=16)
    plt.ylabel(metric_name)#,fontsize=16)
    # plt.ylim(0.35, 0.95)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_loss_history(h):
                
    if isinstance(h, History):
        h = h.history
    
    print(f'\tMin train loss: {np.min(h["loss"])} @epoch {np.argmin(h["loss"])}')
    print(f'\tMin valid loss: {np.min(h["val_loss"])} @epoch {np.argmin(h["val_loss"])}')
    
    # Loss Curves
    plt.figure(figsize=[10,6])
    plt.plot(h['loss'],'ro-')#,linewidth=2.0)
    plt.plot(h['val_loss'],'bo-')#,linewidth=2.0)
    plt.legend(['Training loss', 'Validation loss'])#,fontsize=18)
    # plt.xticks(x, x)
    plt.xlabel('Epoch')#,fontsize=16)
    plt.ylabel('Loss')#,fontsize=16)
    # plt.ylim(0.35, 0.95)
    # plt.title('Loss Curves',fontsize=16)
    plt.show()
          
def plot_custom_metric_history(h, metric_name='bacc'):
    """
    Plot training and validation curves of balanced accuracy metric, for both voting strategies (last, majority).
    """
                
    if isinstance(h, History):
        h = h.history
         
    plt.figure(figsize=[15,6])
    for i, vs in enumerate(voting_strategies):    
        plt.subplot(1, 2, i + 1)
        plt.plot(h[f"train_{vs}_{metric_name}"],'ro-')#,linewidth=2.0)
        plt.plot(h[f"val_{vs}_{metric_name}"],'bo-')#,linewidth=2.0)
        plt.legend([f'Training {metric_name}', f'Validation {metric_name}'])#,fontsize=18)
        # plt.xticks(x, x)
        plt.xlabel('Epoch')#,fontsize=16)
        plt.ylabel(metric_name)#,fontsize=16)
        # plt.ylim(0.35, 0.95)
        plt.title(f'Voting strategy: {vs}') #,fontsize=16)
          
        for dataset_partition in ['train', 'val']:
            print(f'\t[{vs}] Max {dataset_partition} {metric_name}: {np.max(h[f"{dataset_partition}_{vs}_{metric_name}"]):.4f} @epoch {np.argmax(h[f"{dataset_partition}_{vs}_{metric_name}"])}') #, end="")
        print()
    plt.show()

def plot_loss_acc_history(hist):
    
    print(f'\tMin train loss: {np.min(hist.history["loss"])} @epoch {np.argmin(hist.history["loss"])} \t Max train acc: {np.max(hist.history["acc"])} @epoch {np.argmax(hist.history["acc"])}')
    print(f'\tMin valid loss: {np.min(hist.history["val_loss"])} @epoch {np.argmin(hist.history["val_loss"])} \t Max valid acc: {np.max(hist.history["val_acc"])} @epoch {np.argmax(hist.history["val_acc"])}')
    
    # Loss Curves
    plt.figure(figsize=[16,6])
    plt.subplot(121)
    plt.plot(hist.history['loss'],'ro-')#,linewidth=2.0)
    plt.plot(hist.history['val_loss'],'bo-')#,linewidth=2.0)
    plt.legend(['Training loss', 'Validation loss'])#,fontsize=18)
    # plt.xticks(x, x)
    plt.xlabel('Epoch')#,fontsize=16)
    plt.ylabel('Loss')#,fontsize=16)
    # plt.ylim(0.35, 0.95)
    # plt.title('Loss Curves',fontsize=16)
#     plt.show()

    # Accuracy Curves
    plt.subplot(122)
    plt.plot(hist.history['acc'],'ro-')#,linewidth=2.0)
    plt.plot(hist.history['val_acc'],'bo-')#,linewidth=2.0)
    plt.legend(['Training accuracy', 'Validation accuracy'])#,fontsize=18)
    # plt.xticks(x, x)
    plt.xlabel('Epoch')#,fontsize=16)
    plt.ylabel('Accuracy')#,fontsize=16)
    # plt.title('Accuracy Curves',fontsize=16)
    # plt.ylim(0.35, 0.95)
    plt.show()
    
    
def save_train_history(hist, save_path, save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f'{save_name}.pkl'), 'wb') as pickle_filehandler:
        pickle.dump(hist, pickle_filehandler)
          
            
def load_train_history(filepath):   
    with open(filepath, 'rb') as pickle_filehandler:
        hist = pickle.load(pickle_filehandler)   
    print(f'Loading: {filepath.split("/")[-1]}\n')
#     plot_train_history(hist)
    return hist
          