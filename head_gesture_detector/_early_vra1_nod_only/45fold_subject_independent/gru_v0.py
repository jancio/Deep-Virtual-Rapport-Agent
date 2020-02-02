###########################################################
import numpy as np
random_seed = 37
np.random.seed(random_seed)
from tensorflow import set_random_seed
set_random_seed(random_seed)

###########################################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("CUDA_VISIBLE_DEVICES", help="CUDA_VISIBLE_DEVICES", type=int)
parser.add_argument("WINDOW_SIZE", help="WINDOW_SIZE", type=int)
parser.add_argument("N_FEATURES", help="N_FEATURES", type=int)
args = parser.parse_args()
print(args)

###########################################################
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.CUDA_VISIBLE_DEVICES)

###########################################################
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

###########################################################

import time
import glob
from collections import defaultdict
# from keras.layers import Dense, GRU, LSTM, CuDNNGRU, CuDNNLSTM, Activation, LSTMCell, GRUCell, Masking, TimeDistributed, BatchNormalization, Dropout, GaussianNoise
from keras.layers import Dense, GRU, Masking, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from keras.models import Sequential, load_model
from utils import load_train_history, save_train_history, plot_loss_history

# Set parameters
n_folds = 45
WINDOW_SIZE = args.WINDOW_SIZE
N_FEATURES = args.N_FEATURES

gru_arch_range = [
    [64], [128], 
    [32, 32], [64, 64]
]

metrics_names = ['bacc', 'f1', 'precision', 'recall']
voting_strategies = ['last', 'majority']


def arch_to_str(arch):
    return '-'.join(str(n_units) for n_units in arch)


def evaluate_custom_metrics(Y_true, Y_pred, chunk_lens):
    
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

    y_pred['majority'] = np.zeros(len(y_true))
    assert np.sum(chunk_lens) == len(Y_pred)

    chunk_offset = 0
    for chunk_len in chunk_lens:

        # List of lists, each inner list contains votes (0 or 1)
        votes = [[] for _ in range(chunk_len + WINDOW_SIZE - 1)]
        votes_idx = 0

        for segment in Y_pred[chunk_offset:chunk_offset + chunk_len]:
            for pred_label in segment:
                # Append vote (0 or 1)
                votes[votes_idx].append( int(pred_label[0]) )
                votes_idx += 1
            # Reset pointer for the next segment
            votes_idx = votes_idx - WINDOW_SIZE + 1

        # Perform majority voting over votes (skipping the first WINDOW_SIZE - 1)
        for idx, perframe_votes in enumerate(votes[WINDOW_SIZE:]):
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


class CustomMetrics(Callback):
    '''
    e.g. metrics['majority']['f1'] gives list of validation set f1 scores for each epoch 
    '''
    def __init__(self, val_lens):
        self.val_lens = val_lens
        # TODO: ADD BALANCED ACC / WEIGHTED ACC
        self.metrics = dict()
        super(CustomMetrics, self).__init__()
    
    def on_train_begin(self, logs={}):
        for vs in voting_strategies:
            self.metrics[vs] = defaultdict(list)
        
    def on_epoch_end(self, epoch, logs={}):
        val_metrics = evaluate_custom_metrics(Y_true=self.validation_data[1], 
                                              Y_pred=self.model.predict_classes(self.validation_data[0], batch_size=len(self.validation_data[0])), 
                                              chunk_lens=self.val_lens)
    
        # For each voting strategy, calculate all metrics
        for vs in voting_strategies:
            print(f"[{vs}] ", end="")
            for mn in metrics_names:
                self.metrics[vs][mn].append( val_metrics[vs][mn] )
                print(f"- val_{mn}: {self.metrics[vs][mn][-1]:.4f} ", end="")
            print()


# For each architecture
for GRU_ARCH in gru_arch_range:

    dataset_type = f'{WINDOW_SIZE}ws_{N_FEATURES}f'
    dataset_path_prefix = f'/home/ICT2000/jondras/datasets/vra1/subject_independent/{dataset_type}/'

    folds_hist = []
    for k in range(n_folds):

        # Load data for this fold
        data = np.load(dataset_path_prefix + f'{k}fold_{dataset_type}.npz')
        MASK_VALUE = data['MASK_VALUE']   
        
        X_train, Y_train = data['X_train'], data['Y_train']
        X_val,   Y_val   = data['X_val'],   data['Y_val']
        # X_test,  Y_test  = data['X_test'],  data['Y_test']
        
        #######################################################################################################
        # Create model
        
        model = Sequential()
        model.add(Masking(mask_value=MASK_VALUE, input_shape=(WINDOW_SIZE, N_FEATURES)))
        for n_units in GRU_ARCH:
            model.add(GRU(n_units, return_sequences=True, dropout=0.1)) #, recurrent_dropout=0.2))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        if k == 0: 
            print(model.summary())

        #######################################################################################################
        # Train
        
        n_epochs = 100
        batch_size = 128

        # Set early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        # Checkpoint model weights and the model itself: at each epoch
        model_type = f'{k}fold_{arch_to_str(GRU_ARCH)}u_{dataset_type}'
        print(f'INFO:\n\t{model_type}\n\t batch_size={batch_size}\n\t n_epochs={n_epochs}')
        model_checkpoint_path_prefix = f'./checkpoints/{model_type}/'
        if not os.path.exists(model_checkpoint_path_prefix):
            os.makedirs(model_checkpoint_path_prefix)
        model_checkpoint_name = 'm_{epoch:04d}_{loss:.4f}_{val_loss:.4f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_checkpoint_path_prefix + model_checkpoint_name, monitor='val_loss', verbose=1, 
                                           save_best_only=True, save_weights_only=False, mode='auto', period=1)

        # Pass validation chunks' lengths, for this fold
        custom_metrics = CustomMetrics(val_lens=data['val_len'])
        hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=batch_size, epochs=n_epochs, verbose=1, 
                         callbacks=[
                             custom_metrics, 
                             model_checkpoint, 
                             early_stop
                         ], shuffle=True)

        # Plot and save training history
        # plot_loss_history(hist)
        save_train_history(hist, model_type)
        
        #######################################################################################################
        # Save results from this fold
        curr_fold_hist = dict()
        curr_fold_hist.update(hist.history)
        curr_fold_hist.update(custom_metrics.metrics)
        folds_hist.append(curr_fold_hist)

    #     break

    # Save training history from all folds
    save_train_history(folds_hist, f'ALL_{n_folds}fold_{arch_to_str(GRU_ARCH)}u_{dataset_type}')
