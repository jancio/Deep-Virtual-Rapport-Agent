'''
All window sizes: 16, 32, 64
64 GRU units

Running:
python gru_C_v0.py 2 0.45 6
python gru_C_v0.py 2 0.45 12

'''

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
# parser.add_argument("WINDOW_SIZE", help="WINDOW_SIZE", type=int)
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
n_folds = 45
# WINDOW_SIZE = args.WINDOW_SIZE
window_sizes_range = [16, 32, 64]
N_FEATURES = args.N_FEATURES
gru_arch_range = [
    [64], 
    #[128], 
    #[32, 32] #, [64, 64]
]

# For each window size and architecture
for WINDOW_SIZE in window_sizes_range:
    for GRU_ARCH in gru_arch_range:

        dataset_type = f'C_{WINDOW_SIZE}ws_{N_FEATURES}f'
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
            custom_metrics = CustomMetrics(window_size=WINDOW_SIZE, val_lens=data['val_len'])
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
