#######################################################################################################################
# Project: Deep Virtual Rapport Agent (head gesture detector)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Train and validate a final Head Gesture Detector model on the whole 4comb dataset to detect one head gesture
# 
#     Training setting: TS3 (subject-independent dataset split)
#
#     The test set is also used for training (train = train + test, val = val) 
#         => the developed Head Gesture Detector can be evaluated on other datasets (e.g., cross-dataset testing on ccdb)
#            and used for general prediction of head gestures
#
#     Used with all head gestures (nod, shake, tilt)
#         => 3 HGD models were trained.
#
#     Model checkpoints (.hdf) and training history (.pkl) files are saved to the ./checkpoints/final_4comb folder.
#     The best final nod, shake, and tilt HGD models can be found in the folder ./checkpoints/final_4comb/best 
#     Filenames naming convention: {dataset_name}_{head_gesture}_{window_size}ws_{number_of_features}f_{model_architecture}.{hdf5,pkl} 
#     Training history files contain training parameters and train and validation evaluation metrics. 
#
#     Run as:
#         python train_final_4comb_hgd.py {CUDA_VISIBLE_DEVICES} {GPU_MEMORY_FRACTION} {head_gesture}
#                                                                             
#     E.g. 
#         python train_final_4comb_hgd.py 3 0.65 nod
#######################################################################################################################


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
parser.add_argument("GPU_MEMORY_FRACTION", help="GPU_MEMORY_FRACTION", type=float)
parser.add_argument("HEAD_GESTURE", help="HEAD_GESTURE", type=str)
args = parser.parse_args()
print(args)

###########################################################
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.CUDA_VISIBLE_DEVICES)

###########################################################
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.GPU_MEMORY_FRACTION
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
from utils import load_train_history, save_train_history #, plot_loss_history
from utils import evaluate_custom_metrics, CustomMetrics
from utils import voting_strategies, metrics_names, arch_to_str

# Set parameters
WINDOW_SIZE = 32
N_FEATURES = 12
DATASET_NAME = '4comb'
HEAD_GESTURE = args.HEAD_GESTURE

if (HEAD_GESTURE == 'nod') or (HEAD_GESTURE == 'tilt'):
	GRU_ARCH = [16]
elif HEAD_GESTURE == 'shake':
	GRU_ARCH = [8]
else:
	raise Exception(f'{HEAD_GESTURE} is invalid head gesture!')


# dataset_type = f'{DATASET_NAME}_{HEAD_GESTURE}_{WINDOW_SIZE}ws_{N_FEATURES}f'
dataset_type = f'{DATASET_NAME}_{HEAD_GESTURE}_{WINDOW_SIZE}ws_{N_FEATURES}f_A1'
model_type = f'final_{dataset_type}_{arch_to_str(GRU_ARCH)}u'
dataset_path_prefix = f'/home/ICT2000/jondras/dvra_datasets'

# Load data
data = np.load(f'{dataset_path_prefix}/{DATASET_NAME}/segmented_datasets/{dataset_type}.npz')
MASK_VALUE = data['MASK_VALUE']   

X_train, Y_train = data['X_train'], data['Y_train']
X_val,   Y_val   = data['X_val'],   data['Y_val']
X_test,  Y_test  = data['X_test'],  data['Y_test']

# For final training merge train and test partitions
X_train = np.concatenate((X_train, X_test), axis=0)
Y_train = np.concatenate((Y_train, Y_test), axis=0)
del X_test
del Y_test
train_lens = np.concatenate((data['train_len'], data['test_len']), axis=0)

#######################################################################################################
# Create model

model = Sequential()
model.add(Masking(mask_value=MASK_VALUE, input_shape=(WINDOW_SIZE, N_FEATURES)))
for n_units in GRU_ARCH:
    model.add(GRU(n_units, return_sequences=True, dropout=0.1)) #, recurrent_dropout=0.2))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', sample_weight_mode='temporal')
print(model.summary())

#######################################################################################################
# Train

batch_size = 128
n_epochs = 100
# n_epochs = 200
# For augmeted dataset
# n_epochs = 500
# metric_to_monitor = 'val_majority_bacc'
metric_to_monitor = 'val_last_bacc'

# Compute class weights and sample weights; for train and validation sets
train_1_cnt = np.count_nonzero(Y_train[:, -1, 0])
train_01_cnt = float(Y_train.shape[0])
train_class_weights = [train_1_cnt / train_01_cnt, (train_01_cnt - train_1_cnt) / train_01_cnt]
# Assign weights to each training sample
train_sample_weights = np.empty((Y_train.shape[0], WINDOW_SIZE))
for i in range(train_sample_weights.shape[0]):
    for j in range(train_sample_weights.shape[1]):
        train_sample_weights[i, j] = train_class_weights[int(Y_train[i, j, 0])]
        # Equivalent to:
        # train_sample_weights[i, j] = train_class_weights[0] if int(Y_train[i, j, 0]) == 0 else train_class_weights[1]
        
val_1_cnt = np.count_nonzero(Y_val[:, -1, 0])
val_01_cnt = float(Y_val.shape[0])
val_class_weights = [val_1_cnt / val_01_cnt, (val_01_cnt - val_1_cnt) / val_01_cnt]
# Assign weights to each validation sample
val_sample_weights = np.empty((Y_val.shape[0], WINDOW_SIZE))
for i in range(val_sample_weights.shape[0]):
    for j in range(val_sample_weights.shape[1]):
        val_sample_weights[i, j] = val_class_weights[int(Y_val[i, j, 0])]
        
        
# Set callback for custom metrics (pass validation chunks' lengths)
custom_metrics = CustomMetrics(window_size=WINDOW_SIZE, training_data=(X_train, Y_train), 
                               train_lens=train_lens, val_lens=data['val_len'])

# Checkpoint model weights and the model itself, if improved
model_checkpoint_path_prefix = f'./checkpoints/final_4comb/'
if not os.path.exists(model_checkpoint_path_prefix):
    os.makedirs(model_checkpoint_path_prefix)
model_checkpoint = ModelCheckpoint(model_checkpoint_path_prefix + f'{model_type}.hdf5', monitor=metric_to_monitor, verbose=1, 
                                   save_best_only=True, save_weights_only=False, mode='max', period=1)

# Set early stopping
# early_stop = EarlyStopping(monitor=metric_to_monitor, patience=10, verbose=1, mode='max', restore_best_weights=True)

print(f'INFO:\n\t{model_type}\n\t batch_size={batch_size}\n\t n_epochs={n_epochs}\n\t metric_to_monitor={metric_to_monitor}\n\t train_class_weights={train_class_weights}\n\t val_class_weights={val_class_weights}\n\t #params={model.count_params()}')
hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val, val_sample_weights), batch_size=batch_size, epochs=n_epochs, 
                 verbose=1, 
                 sample_weight=train_sample_weights, 
                 callbacks=[
                     custom_metrics, 
                     model_checkpoint, 
                     # early_stop
                 ], shuffle=True)

#######################################################################################################
# Save training history
# print(hist.history)
hist.history['model_type'] = model_type
hist.history['batch_size'] = batch_size
hist.history['n_epochs'] = n_epochs
hist.history['metric_to_monitor'] = metric_to_monitor
hist.history['train_class_weights'] = train_class_weights
hist.history['val_class_weights'] = val_class_weights
hist.history['n_model_params'] = model.count_params()
save_train_history(hist, model_checkpoint_path_prefix, f'{model_type}')
