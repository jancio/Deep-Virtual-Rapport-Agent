#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Final training of rapport models on the whole dataset, for online prediction.
#     No hold-out test set created.
#     No cross-validation.
#     Single-split validation is either subject-dependent or subject-independent.
#     Otherwise, based on train.py.
#######################################################################################################################

import os
import sys
import time
import shutil
import logging
from logger import DualLogger
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut

import torch
from torch.utils.tensorboard import SummaryWriter

from data import get_unimodal_base_dataset_loaders
from data import get_multimodal_base_dataset_loaders

from model import UnimodalBaseClassifier
from model import UnimodalTCNClassifier
from model import MultimodalBaseClassifier
from model import MultimodalTCNClassifier

from colors import ConsoleColors

from utils import time_diff
from utils import calculate_metrics

from constants import all_labels_names_set
from constants import minimize_metrics

# Reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def main():
    parser = argparse.ArgumentParser()

    # Names, paths, logs
    parser.add_argument('--dataset_path', default=f'/home/ICT2000/jondras/datasets/mimicry/segmented_datasets',
                        help='path prefix to dataset directory (excludes the suffix with dataset version e.g. "_v0")')
    parser.add_argument('--dataset_version', default=f'v2', help='version of the dataset (v0|v1|v2|v3)')
    parser.add_argument('--logger_path', default='/home/ICT2000/jondras/deep-virtual-rapport-agent/rapport_model/logs',
                        help='path to logging directory (will be created)')
    parser.add_argument('--logger_dir_suffix', default='final', help='suffix of the log directory')

    # Data parameters
    parser.add_argument('--speech_feature_type', default='emobase', help='emobase|mfcc')
    parser.add_argument('--workers_num', default=1, type=int, help='number of workers for data loading')
    parser.add_argument('--labels_names', default=['nod',
                                                   'shake',
                                                   'tilt',
                                                   'smile',
                                                   'gaze_away',
                                                   'voice_active',
                                                   'take_turn'], nargs='+',
                        help='names of target labels (will be predicted in this order)')
    parser.add_argument('--sequence_length', default=8, type=int,
                        help='maximum length of feature sequences (i.e. window size)')
    parser.add_argument('--modality', default='multimodal', help='speech|vision|multimodal')

    # Model parameters
    parser.add_argument('--model_type', default='multimodal-base-classifier',
                        help='unimodal-base-classifier|unimodal-tcn-classifier|multimodal-base-classifier|'
                             'multimodal-tcn-classifier')
    parser.add_argument('--vision_rnn_layer_dim', default=32, type=int, help='dimensionality of vision RNN/TCN layers')
    parser.add_argument('--speech_rnn_layer_dim', default=64, type=int, help='dimensionality of speech RNN/TCN layers')
    parser.add_argument('--rnn_layer_num', default=1, type=int,
                        help='number of RNN/TCN layers (same for each submodel of a multimodal model)')
    parser.add_argument('--bidirectional', default=False, help='bidirectional RNN (embedding_dim will be halved)')
    parser.add_argument('--rnn_dropout_rate', default=.1,
                        help='dropout rate after each RNN/TCN layer (except the last layer)')
    parser.add_argument('--tcn_kernel_size', default=5, type=int, help='kernel size for TCN models')
    parser.add_argument('--fc_layer_num', default=64, type=int, help='number of fully-connected layers after RNN/TCN')
    parser.add_argument('--fc_dropout_rate', default=.1,
                        help='dropout rate after each fully-connected layer (except the last layer)')

    # Training and optimization
    parser.add_argument('--validation_set_size', default=0.2, help='fraction of the training set used for validation')
    parser.add_argument('--subject_independent_validation', default=False,
                        help='perform (single-split) subject-independent validation')
    parser.add_argument('--epochs_num', default=10, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='size of a mini-batch')
    parser.add_argument('--weight_decay', default=.0, help='decay (l2 norm) for the optimizer weights')
    parser.add_argument('--learning_rate', default=.0001, help='initial learning rate')
    parser.add_argument('--monitored_metric', default='loss',
                        help='evaluation metric monitored during training and used to checkpoint the best model '
                             '(loss|bacc|f1|precision|recall)')
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of a GPU to use (0|1|2|3)')

    opt = parser.parse_args()

    # Add derived/additional options
    opt.dataset_path = f'{opt.dataset_path}_{opt.dataset_version}'
    # Path to the dataset file with metadata and labels for each sequence
    opt.dataset_file_path = os.path.join(opt.dataset_path, f'metadata_labels_{opt.sequence_length}ws.csv')
    opt.features_path = {
        'vision': os.path.join(opt.dataset_path, 'vision_features', f'{opt.sequence_length}ws'),
        'speech': os.path.join(opt.dataset_path, 'audio_features', opt.speech_feature_type, f'{opt.sequence_length}ws')
    }
    opt.logger_path = os.path.join(opt.logger_path,
                                   f'{int(time.time())}_{opt.modality}_{opt.model_type}_{opt.logger_dir_suffix}')
    opt.class_num = len(opt.labels_names)
    assert set(opt.labels_names) <= all_labels_names_set, f'Labels names {opt.labels_names} must form a subset of ' \
                                                          f'{all_labels_names_set}!'
    # Dataset partitions that will be generated
    opt.dataset_partitions = ['train', 'val']
    # Set feature dimensionality based on the dataset version
    if opt.dataset_version == 'v0':
        opt.vision_feature_dim = 11
        opt.speech_feature_dim = 52
    elif opt.dataset_version == 'v1':
        opt.vision_feature_dim = 12
        opt.speech_feature_dim = 53
    elif opt.dataset_version == 'v2' or opt.dataset_version == 'v3':
        opt.vision_feature_dim = 27
        opt.speech_feature_dim = 53
    else:
        raise Exception('The version of the dataset needs to be one of v0|v1|v2|v3!')

    # Use the specified GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    torch.cuda.set_device(int(opt.gpu_id))

    # Set up stdout logger (for stdout and stderr)
    os.makedirs(opt.logger_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(opt.logger_path, f'final_train.log'),
                        level=logging.INFO, format='%(asctime)s %(levelname)s ==> %(message)s')
    sys.stdout = DualLogger('stdout')
    sys.stderr = DualLogger('stderr')
    # Tensorboard logger
    tensorboard_logger = SummaryWriter(log_dir=opt.logger_path, flush_secs=5)

    # Print all args/options/settings
    print(f'{ConsoleColors.CC_GREY}\nTraining and validating models{ConsoleColors.CC_END}')
    for arg in vars(opt):
        print(f'{arg}={ConsoleColors.CC_YELLOW}{str(getattr(opt, arg))}{ConsoleColors.CC_END}')

    # Read metadata+labels file (i.e., dataset file)
    metadata_labels = pd.read_csv(opt.dataset_file_path)#[:1000]
    sequence_ids = metadata_labels['sequence_id'].tolist()

    all_idx = [idx for idx, _ in enumerate(sequence_ids)]
    val_len = int(len(all_idx) * opt.validation_set_size)

    # Perform (single-split) subject-independent validation: number of subjects in the validation set is given by the
    # validation_set_size (so that the actual validation set size will be at least the validation_set_size)
    if opt.subject_independent_validation:
        # Use listener subject id for subject-independent (leave-one-subject-out (LOSO)) cross-validation, since the
        # listener is the target subject
        subject_ids = metadata_labels['listener_sid'].tolist()
        cross_validator = LeaveOneGroupOut()
        cross_validator_splits = cross_validator.split(sequence_ids, groups=subject_ids)
        val_idx = []
        for val_subject_cnt, (_, one_subject_val_idx) in enumerate(cross_validator_splits):
            val_idx.extend(one_subject_val_idx)
            if len(val_idx) >= val_len:
                break
        val_subject_cnt += 1
        train_idx = list(set(all_idx) - set(val_idx))
        print(f'Validation subjects: {val_subject_cnt}\t Actual validation set size: {len(val_idx) / len(all_idx):.2f}')

    # Otherwise, perform (single-split) subject-dependent validation
    else:
        train_idx, val_idx = train_test_split(all_idx, test_size=opt.validation_set_size, shuffle=True)

    ids = {}
    ids['train'] = [sequence_ids[x] for x in train_idx]
    ids['val'] = [sequence_ids[x] for x in val_idx]

    # Data loaders
    if opt.modality == 'speech' or opt.modality == 'vision':
        if opt.model_type == 'unimodal-base-classifier' or opt.model_type == 'unimodal-tcn-classifier':
            loaders = get_unimodal_base_dataset_loaders(metadata_labels=metadata_labels, ids=ids, opt=opt)
        else:
            print(f'Data loader is not implemented for {opt.modality} -> {opt.model_type}')

    elif opt.modality == 'multimodal':
        if opt.model_type == 'multimodal-base-classifier' or opt.model_type == 'multimodal-tcn-classifier':
            loaders = get_multimodal_base_dataset_loaders(metadata_labels=metadata_labels, ids=ids, opt=opt)
        else:
            print(f'Data loader is not implemented for {opt.modality} -> {opt.model_type}')

    else:
        print(f'Data loader is not implemented for {opt.modality}')

    # Model, optimizer and loss function
    if opt.modality == 'speech' or opt.modality == 'vision':
        if opt.model_type == 'unimodal-base-classifier':
            model = UnimodalBaseClassifier(opt=opt).cuda()
            optimizer = torch.optim.Adam(model.optim_params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
                                         amsgrad=True)
            loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')

        elif opt.model_type == 'unimodal-tcn-classifier':
            model = UnimodalTCNClassifier(opt=opt).cuda()
            optimizer = torch.optim.Adam(model.optim_params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
                                         amsgrad=True)
            loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            print(f'Model is not implemented for {opt.modality} -> {opt.model_type}')

    elif opt.modality == 'multimodal':
        if opt.model_type == 'multimodal-base-classifier':
            model = MultimodalBaseClassifier(opt=opt).cuda()
            optimizer = torch.optim.Adam(model.optim_params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
                                         amsgrad=True)
            loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')

        elif opt.model_type == 'multimodal-tcn-classifier':
            model = MultimodalTCNClassifier(opt=opt).cuda()
            optimizer = torch.optim.Adam(model.optim_params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
                                         amsgrad=True)
            loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            print(f'Model is not implemented for {opt.modality} -> {opt.model_type}')

    else:
        print(f'Model is not implemented for {opt.modality}')

    # Set up logging
    logger_path = os.path.join(opt.logger_path, f'final')
    os.makedirs(logger_path, exist_ok=True)
    checkpoint_file_name = os.path.join(logger_path, 'checkpoint.pth.tar')
    best_model_file_name = os.path.join(logger_path, 'model_best.pth.tar')

    print(f'{ConsoleColors.CC_GREY}{model}{ConsoleColors.CC_END}')
    best_monitored_metric = sys.float_info.max if opt.monitored_metric in minimize_metrics else sys.float_info.min

    # Train and validate
    start_time = time.time()
    for epoch in range(1, opt.epochs_num + 1):
        print(f'\n{ConsoleColors.CC_BOLD}{ConsoleColors.CC_RED}epoch: {ConsoleColors.CC_END} {epoch}/{opt.epochs_num}')

        train_metrics = train_classifier(loaders['train'], model, optimizer, loss_function, opt.labels_names)
        val_metrics = validate_classifier(loaders['val'], model, loss_function, opt.labels_names)
        metrics = {
            'train': train_metrics,
            'val': val_metrics
        }
        print(f"{ConsoleColors.CC_BOLD}\t\t train_loss: {ConsoleColors.CC_END}{metrics['train']['overall']['loss']:.5f}"
              f" {ConsoleColors.CC_BOLD}\t val_loss: {ConsoleColors.CC_END}{metrics['val']['overall']['loss']:.5f}",
              end="")

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_monitored_metric': (opt.monitored_metric,
                                      min(best_monitored_metric, metrics['val']['overall'][opt.monitored_metric])
                                      if opt.monitored_metric in minimize_metrics
                                      else max(best_monitored_metric, metrics['val']['overall'][opt.monitored_metric])),
            'opt': opt,
            'metrics': metrics
        }
        torch.save(state, checkpoint_file_name)

        is_best = (((opt.monitored_metric in minimize_metrics)
                    and (metrics['val']['overall'][opt.monitored_metric] < best_monitored_metric))
                   or ((opt.monitored_metric not in minimize_metrics)
                       and (metrics['val']['overall'][opt.monitored_metric] > best_monitored_metric)))
        if is_best:
            # Update the best checkpoint and metrics
            shutil.copyfile(checkpoint_file_name, best_model_file_name)
            best_monitored_metric = metrics['val']['overall'][opt.monitored_metric]
            print(f'\t{ConsoleColors.CC_BEIGE}best model improved{ConsoleColors.CC_END}', end='')

        # Log all metrics to tensorboard
        for dataset_partition in metrics.keys():
            for label_name in metrics[dataset_partition].keys():
                for metric_name in metrics[dataset_partition][label_name].keys():
                    tensorboard_logger.add_scalar(
                        f'final/{dataset_partition}/{label_name}/{metric_name}',
                        metrics[dataset_partition][label_name][metric_name], epoch)

    print(f'\n    total time: {ConsoleColors.CC_YELLOW2}{time_diff(start_time)}{ConsoleColors.CC_END}')

    tensorboard_logger.close()


def train_classifier(train_loader, model, optimizer, loss_function, labels_names):
    """Train the model on all training batches from one epoch.

    Same train function is used to train the base and TCN classifier.
    Same train function is used to train unimodal and multimodal models.

    Args:
        train_loader: PyTorch train data loader
        model: PyTorch model
        optimizer: PyTorch optimizer
        loss_function: PyTorch loss function
        labels_names (array/list): Names of labels/targets

    Returns:
        train_metrics (dict of dict of lists): Dictionary that maps class/label/target name and metric name to a
            corresponding training metric value. Besides the class/label/target names provided in labels_names, it also
            contains a key 'overall' referring to the average over all targets/labels.
    """

    # Aggregates from all training batches from one epoch
    all_predictions = []
    all_labels = []
    all_losses = []

    # Apply loss weights based on the train dataset class weights
    loss_function.pos_weight = train_loader.dataset.pos_class_weights.cuda()

    for i, train_data in enumerate(tqdm(train_loader, desc='    train')):
        features, labels = train_data

        optimizer.zero_grad()

        features = [f.cuda() for f in features]
        labels = labels.cuda()

        logits, predictions = model.forward(*features)

        # Get (already weighted) loss matrix (batch_size x class_num)
        batch_losses = loss_function(logits, labels)

        # Reduce/average loss for each class
        # In case of reduction='none', the loss needs to be normalized manually
        pos_labels_cnt = labels.sum(dim=0)
        neg_labels_cnt = len(labels) - pos_labels_cnt
        class_losses = batch_losses.sum(dim=0) / (loss_function.pos_weight * pos_labels_cnt + 1. * neg_labels_cnt)

        loss = class_losses.mean()
        loss.backward()
        optimizer.step()

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())
        all_losses.extend(batch_losses.tolist())

    # Calculate training metrics
    train_metrics = calculate_metrics(all_predictions, all_labels, labels_names, losses=all_losses,
                                      loss_function_pos_weights=loss_function.pos_weight.tolist())

    return train_metrics


def validate_classifier(val_loader, model, loss_function, labels_names):
    """Validate the model on all validation batches from one epoch.

    Same validation function is used to validate the base and TCN classifier.
    Same validation function is used to validate unimodal and multimodal models.

    Args:
        val_loader: PyTorch validation data loader
        model: PyTorch model
        loss_function: PyTorch loss function
        labels_names (array/list): Names of labels/targets

    Returns:
        val_metrics (dict of dict of lists): Dictionary that maps class/label/target name and metric name to a
            corresponding validation metric value. Besides the class/label/target names provided in labels_names, it
            also contains a key 'overall' referring to the average over all targets/labels.
    """

    with torch.no_grad():
        all_predictions = []
        all_labels = []
        all_losses = []

        # Apply loss weights based on validation dataset class weights
        loss_function.pos_weight = val_loader.dataset.pos_class_weights.cuda()

        for i, val_data in enumerate(tqdm(val_loader, desc='    valid')):
            features, labels = val_data

            features = [f.cuda() for f in features]
            labels = labels.cuda()

            logits, predictions = model.forward(*features)

            # Get (already weighted) loss matrix (batch_size x class_num)
            batch_losses = loss_function(logits, labels)

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            all_losses.extend(batch_losses.tolist())

        # Calculate validation metrics
        val_metrics = calculate_metrics(all_predictions, all_labels, labels_names, losses=all_losses,
                                        loss_function_pos_weights=loss_function.pos_weight.tolist())

    return val_metrics


if __name__ == '__main__':
    main()
