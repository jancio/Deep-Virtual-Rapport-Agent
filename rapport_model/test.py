#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Test rapport models on the hold-out test sets from all folds (from either subject-dependent or subject-independent
# cross-validation).
#     Logs from testing are saved in the same logging directory as provided in the logger_path argument.
#######################################################################################################################

import os
import sys
import time
import logging
from logger import DualLogger
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut

import torch

from data import get_unimodal_base_dataset_loaders
from data import get_multimodal_base_dataset_loaders

from model import UnimodalBaseClassifier
from model import UnimodalTCNClassifier
from model import MultimodalBaseClassifier
from model import MultimodalTCNClassifier

from utils import time_diff
from utils import calculate_metrics

from colors import ConsoleColors

# Reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# Use GPU
device = torch.device('cuda')


def main():
    parser = argparse.ArgumentParser()
    # Names, paths, logs
    parser.add_argument('--dataset_path', default=f'/home/ICT2000/jondras/datasets/mimicry/segmented_datasets',
                        help='path prefix to dataset directory (excludes the suffix with dataset version e.g. "_v0")')
    parser.add_argument('--dataset_version', default=f'v3', help='version of the dataset (v0|v1|v2|v3)')
    parser.add_argument('--logger_path',
                        default='/home/ICT2000/jondras/deep-virtual-rapport-agent/rapport_model/logs/1569294550_multimodal_multimodal-base-classifier_nod',
                        help='path to logging directory containing the final model')
    # Data parameters
    parser.add_argument('--sequence_length', default=32, type=int,
                        help='maximum length of feature sequences (i.e. window size)')
    # Training and optimization
    parser.add_argument('--loso_cross_validation', default=False,
                        help='load model from subject-independent (leave-one-subject-out (LOSO)) cross-validation')
    parser.add_argument('--fold_num', default=10, type=int,
                        help='number of folds, relevant only for subject-dependent cross-validation')
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of a GPU to use (0|1|2|3)')
    opt = parser.parse_args()

    # Add derived/additional options
    opt.dataset_path = f'{opt.dataset_path}_{opt.dataset_version}'
    # Path to the dataset file with metadata and labels for each sequence
    opt.dataset_file_path = os.path.join(opt.dataset_path, f'metadata_labels_{opt.sequence_length}ws.csv')

    # Use the specified GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    torch.cuda.set_device(int(opt.gpu_id))

    # Set up stdout logger (for stdout and stderr)
    os.makedirs(opt.logger_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(opt.logger_path, f'test.log'),
                        level=logging.INFO, format='%(asctime)s %(levelname)s ==> %(message)s')
    sys.stdout = DualLogger('stdout')
    sys.stderr = DualLogger('stderr')

    # Print all args/options/settings
    print(f'{ConsoleColors.CC_GREY}\nTraining and validating models{ConsoleColors.CC_END}')
    for arg in vars(opt):
        print(f'{arg}={ConsoleColors.CC_YELLOW}{str(getattr(opt, arg))}{ConsoleColors.CC_END}')

    # Read metadata+labels file
    metadata_labels = pd.read_csv(opt.dataset_file_path)#[:100]
    sequence_ids = metadata_labels['sequence_id'].tolist()

    ids = {}
    fold = 0

    # Load model from subject-independent (leave-one-subject-out (LOSO)) cross-validation
    if opt.loso_cross_validation:
        # Use listener subject id for subject-independent (leave-one-subject-out (LOSO)) cross-validation, since the
        # listener is the target subject
        subject_ids = metadata_labels['listener_sid'].tolist()
        opt.fold_num = len(np.unique(subject_ids))
        cross_validator = LeaveOneGroupOut()
        cross_validator_splits = cross_validator.split(sequence_ids, groups=subject_ids)
    # Otherwise, load model from subject-dependent k-fold cross-validation
    else:
        cross_validator = KFold(n_splits=opt.fold_num, shuffle=True)
        cross_validator_splits = cross_validator.split(sequence_ids)

    metrics = dict()

    # Cross-validation loop: test on each fold
    for _, test_idx in cross_validator_splits:
        fold_start_time = time.time()
        print(f'\n{ConsoleColors.CC_BOLD}{ConsoleColors.CC_GREEN}fold:{ConsoleColors.CC_END} {fold + 1}/{opt.fold_num}')

        ids['test'] = [sequence_ids[x] for x in test_idx]

        # Checkpoint
        checkpoint = torch.load(os.path.join(opt.logger_path, f'fold_{fold + 1:02d}', 'model_best.pth.tar'),
                                map_location=device)
        print(f'\t{ConsoleColors.CC_BOLD}best_epoch:{ConsoleColors.CC_END} {checkpoint["epoch"]} \t '
              f'{ConsoleColors.CC_BOLD}best_monitored_metric:{ConsoleColors.CC_END} '
              f'{checkpoint["best_monitored_metric"]}')
        checkpoint_opt = checkpoint['opt']
        # Dataset partitions that will be generated
        checkpoint_opt.dataset_partitions = ['test']

        # Check whether checkpoint options agree with the current (specified) options
        assert checkpoint_opt.sequence_length == opt.sequence_length, f'The specified sequence length does not match the checkpoint one ({checkpoint_opt.sequence_length})!'
        assert checkpoint_opt.loso_cross_validation == opt.loso_cross_validation, f'The specified type of crossvalidation does not match the checkpoint one ({checkpoint_opt.loso_cross_validation})!'
        assert checkpoint_opt.fold_num == opt.fold_num, f'The specified number of folds does not match the checkpoint one ({checkpoint_opt.fold_num})!'

        # Data loaders
        if checkpoint_opt.modality == 'speech' or checkpoint_opt.modality == 'vision':
            if checkpoint_opt.model_type == 'unimodal-base-classifier' \
                    or checkpoint_opt.model_type == 'unimodal-tcn-classifier':
                loaders = get_unimodal_base_dataset_loaders(metadata_labels=metadata_labels, ids=ids,
                                                            opt=checkpoint_opt)
            else:
                print(f'Data loader is not implemented for {checkpoint_opt.modality} -> {checkpoint_opt.model_type}')

        elif checkpoint_opt.modality == 'multimodal':
            if checkpoint_opt.model_type == 'multimodal-base-classifier' \
                    or checkpoint_opt.model_type == 'multimodal-tcn-classifier':
                loaders = get_multimodal_base_dataset_loaders(metadata_labels=metadata_labels, ids=ids,
                                                              opt=checkpoint_opt)
            else:
                print(f'Data loader is not implemented for {checkpoint_opt.modality} -> {checkpoint_opt.model_type}')

        else:
            print(f'Data loader is not implemented for {checkpoint_opt.modality}')

        # Model
        if checkpoint_opt.modality == 'speech' or checkpoint_opt.modality == 'vision':
            if checkpoint_opt.model_type == 'unimodal-base-classifier':
                model = UnimodalBaseClassifier(opt=checkpoint_opt).cuda()
            elif checkpoint_opt.model_type == 'unimodal-tcn-classifier':
                model = UnimodalTCNClassifier(opt=checkpoint_opt).cuda()
            else:
                print(f'Model is not implemented for {checkpoint_opt.modality} -> {checkpoint_opt.model_type}')

        elif checkpoint_opt.modality == 'multimodal':
            if checkpoint_opt.model_type == 'multimodal-base-classifier':
                model = MultimodalBaseClassifier(opt=checkpoint_opt).cuda()
            elif checkpoint_opt.model_type == 'multimodal-tcn-classifier':
                model = MultimodalTCNClassifier(opt=checkpoint_opt).cuda()
            else:
                print(f'Model is not implemented for {checkpoint_opt.modality} -> {checkpoint_opt.model_type}')

        else:
            print(f'Model is not implemented for {checkpoint_opt.modality}')
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # Get test metrics for this fold
        fold_metrics = test_classifier(loaders['test'], model, checkpoint_opt.labels_names)
        for label_name in fold_metrics.keys():
            if label_name not in metrics:
                metrics[label_name] = defaultdict(list)
            for metric_name in fold_metrics[label_name].keys():
                metrics[label_name][metric_name].append(fold_metrics[label_name][metric_name])

        fold += 1
        print(f'    fold time: {ConsoleColors.CC_YELLOW2}{time_diff(fold_start_time)}{ConsoleColors.CC_END}')

    # Calculate metrics over all folds
    for label_name in metrics.keys():
        print()
        for metric_name in metrics[label_name].keys():
            metrics[label_name][metric_name] = {
                'mean': np.mean(metrics[label_name][metric_name]),
                'std': np.std(metrics[label_name][metric_name])
            }
            print(f'- {label_name}\t- {metric_name}:\t {metrics[label_name][metric_name]["mean"]:.4f} '
                  f'+/- {metrics[label_name][metric_name]["std"]:.4f}')
    # Can also save metrics to pickle (but not needed since everything is in test.log)


def test_classifier(test_loader, model, labels_names):
    """Test the model on all test batches.

    Same test function is used to test the base and TCN classifier.
    Same test function is used to test unimodal and multimodal models.

    Args:
        test_loader: PyTorch test data loader
        model: PyTorch model
        labels_names (array/list): Names of labels/targets

    Returns:
        test_metrics (dict of dict of lists): Dictionary that maps class/label/target name and metric name to a
            corresponding test metric value. Besides the class/label/target names provided in labels_names, it also
            contains a key 'overall' referring to the average over all targets/labels.
    """

    with torch.no_grad():
        # Aggregates from all test batches
        all_predictions = []
        all_labels = []

        for i, test_data in enumerate(tqdm(test_loader, desc='    test')):
            features, labels = test_data

            features = [f.cuda() for f in features]
            labels = labels.cuda()

            _, predictions = model.forward(*features)

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

        # Calculate test metrics
        test_metrics = calculate_metrics(all_predictions, all_labels, labels_names)

    return test_metrics


if __name__ == '__main__':
    main()
