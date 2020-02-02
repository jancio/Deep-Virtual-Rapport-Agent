#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# [Currently, just a skeleton code.]
# Online prediction using rapport models.
#     For now, to simulate the online prediction, there is an extra code within the
#     "ONLY FOR SIMULATION OF ONLINE PREDICTION" blocks.
#     Logs from online prediction are saved in the same logging directory as provided in the logger_path argument.
#######################################################################################################################

import os
import sys
import glob
import time
import logging
from logger import DualLogger
import argparse
import pandas as pd
import numpy as np
from collections import deque
import torch

from data import get_unimodal_base_dataset_loaders
from data import get_multimodal_base_dataset_loaders

from model import UnimodalBaseClassifier
from model import UnimodalTCNClassifier
from model import MultimodalBaseClassifier
from model import MultimodalTCNClassifier

from colors import ConsoleColors

from constants import vision_features_names
from constants import vision_features_names_to_normalize
from constants import vision_features_names_to_diff
from constants import speech_features_names


# Reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# Use GPU
device = torch.device('cuda')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logger_path',
                        default='/home/ICT2000/jondras/deep-virtual-rapport-agent/rapport_model/logs/1569291742_multimodal_multimodal-base-classifier_nod',
                        help='path to logging directory containing the final model')
    parser.add_argument('--predictions_buffer_size', default=10, type=int, help='number of past predictions to buffer')
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of a GPU to use (0|1|2|3)')
    # Normalization parameters for online prediction
    parser.add_argument('--normalization_params_dir',
                        default='/home/ICT2000/jondras/deep-virtual-rapport-agent/rapport_model/normalization_params',
                        help='path to normalization parameters directory for online prediction')
    parser.add_argument('--normalization_params_filename',
                        default='',
                        help='(optional) name of a specific file with normalization parameters; if not provided, the '
                             'file with the latest timestep will be used')
    opt = parser.parse_args()

    # Use the specified GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    torch.cuda.set_device(int(opt.gpu_id))

    # Set up stdout logger (for stdout and stderr)
    os.makedirs(opt.logger_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(opt.logger_path, f'predict_online.log'),
                        level=logging.INFO, format='%(asctime)s %(levelname)s ==> %(message)s')
    sys.stdout = DualLogger('stdout')
    sys.stderr = DualLogger('stderr')

    # Print args/options/settings
    print(f'{ConsoleColors.CC_GREY}\nPredicting online{ConsoleColors.CC_END}')
    for arg in vars(opt):
        print(f'{arg}={ConsoleColors.CC_YELLOW}{str(getattr(opt, arg))}{ConsoleColors.CC_END}')

    # Load checkpoint
    checkpoint = torch.load(os.path.join(opt.logger_path, f'final', 'model_best.pth.tar'), map_location=device)
    print(f'\t{ConsoleColors.CC_BOLD}best_epoch:{ConsoleColors.CC_END} {checkpoint["epoch"]} \t {ConsoleColors.CC_BOLD}'
          f'best_monitored_metric:{ConsoleColors.CC_END} {checkpoint["best_monitored_metric"]}\n')
    checkpoint_opt = checkpoint['opt']
    # Set batch size for online prediction
    checkpoint_opt.batch_size = 1
    # Set features names based on the dataset version and speech feature type
    vision_features_ordered_names = vision_features_names[checkpoint_opt.dataset_version]
    speech_features_ordered_names = speech_features_names[checkpoint_opt.dataset_version][checkpoint_opt.speech_feature_type]

    # Setup model
    if checkpoint_opt.modality == 'speech' or checkpoint_opt.modality == 'vision':
        if checkpoint_opt.model_type == 'unimodal-base-classifier':
            model = UnimodalBaseClassifier(opt=checkpoint_opt).cuda()
        elif opt.model_type == 'unimodal-tcn-classifier':
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
    # Initialize RNN hidden state to zeros (done manually for online prediction)
    model.init_hidden()

    # Load normalization parameters
    # If a specific filename is not provided, take the normalization parameters from the file with the latest timestamp
    if opt.normalization_params_filename == '':
        normalization_params_filename = sorted(glob.glob(os.path.join(opt.normalization_params_dir, '*.npz')))[-1]
    else:
        normalization_params_filename = os.path.join(opt.normalization_params_dir, opt.normalization_params_filename)
    print(f'Loading normalization parameters from: {ConsoleColors.CC_YELLOW}{normalization_params_filename}'
          f'{ConsoleColors.CC_END}\n')
    normalization_params = np.load(normalization_params_filename)

    # ############################ ONLY FOR SIMULATION OF ONLINE PREDICTION - BLOCK START ##############################
    # Load data
    sequence_length = 32
    dataset_path = '/home/ICT2000/jondras/datasets/mimicry/segmented_datasets_v3'
    dataset_file_path = os.path.join(dataset_path, f'metadata_labels_{sequence_length}ws.csv')
    metadata_labels = pd.read_csv(dataset_file_path)
    metadata_labels = metadata_labels[:1000]

    sequence_ids = metadata_labels['sequence_id'].tolist()
    ids = dict(test=sequence_ids)
    checkpoint_opt.dataset_partitions = ['test']

    if checkpoint_opt.modality == 'speech' or checkpoint_opt.modality == 'vision':
        if checkpoint_opt.model_type == 'unimodal-base-classifier' \
                or checkpoint_opt.model_type == 'unimodal-tcn-classifier':
            loaders = get_unimodal_base_dataset_loaders(metadata_labels=metadata_labels, ids=ids, opt=checkpoint_opt)
        else:
            print(f'Data loader is not implemented for {checkpoint_opt.modality} -> {checkpoint_opt.model_type}')
    elif checkpoint_opt.modality == 'multimodal':
        if checkpoint_opt.model_type == 'multimodal-base-classifier' \
                or checkpoint_opt.model_type == 'multimodal-tcn-classifier':
            loaders = get_multimodal_base_dataset_loaders(metadata_labels=metadata_labels, ids=ids, opt=checkpoint_opt)
        else:
            print(f'Data loader is not implemented for {checkpoint_opt.modality} -> {checkpoint_opt.model_type}')
    else:
        print(f'Data loader is not implemented for {opt.modality}')

    data_iterator = iter(loaders['test'])
    # ############################ ONLY FOR SIMULATION OF ONLINE PREDICTION - BLOCK END ################################

    # Main loop
    predictions_buffer = deque(maxlen=opt.predictions_buffer_size)
    with torch.no_grad():

        while True:
            # ############################ ONLY FOR SIMULATION OF ONLINE PREDICTION - BLOCK START ######################
            # Simulate time delays (exaggerated)
            time.sleep(2.)
            # Get features
            features, _ = next(data_iterator)
            [vision_features, speech_features] = features
            # Get feature vectors as a single batch and for the last timestep only
            vision_features = vision_features[0, -1].view(1, 1, -1)
            speech_features = speech_features[0, -1].view(1, 1, -1)
            print(vision_features.shape, speech_features.shape)
            # Create dictionaries mapping feature names to features
            # Need to artificially add raw features that would be provided by OpenFace in a non-simulation setting
            vision_features_dict = {
                ' pose_Tx': np.random.random(),
                ' pose_Ty': np.random.random(),
                ' pose_Tz': np.random.random(),
                ' pose_Rx': np.random.random(),
                ' pose_Ry': np.random.random(),
                ' pose_Rz': np.random.random(),
                ' gaze_angle_x': np.random.random(),
                ' gaze_angle_y': np.random.random()
            }
            speech_features_dict = dict()
            for i, feature_name in enumerate(vision_features_ordered_names):
                vision_features_dict[feature_name] = vision_features[0, 0, i]
            for i, feature_name in enumerate(speech_features_ordered_names):
                speech_features_dict[feature_name] = speech_features[0, 0, i]
            # ############################ ONLY FOR SIMULATION OF ONLINE PREDICTION - BLOCK END ########################

            # Get vision and speech features, from OpenFace and OpenSmile respectively
            # ... TODO
            # vision_features_dict =
            # speech_features_dict =

            # Preprocess features (first order differences and normalization)
            if not predictions_buffer:
                previous_vision_features_dict = vision_features_dict
            vision_features_dict = preprocess_vision_features(vision_features_dict, previous_vision_features_dict,
                                                              normalization_params)
            previous_vision_features_dict = vision_features_dict
            speech_features_dict = preprocess_speech_features(speech_features_dict)
            # Convert features dictionaries to tensors of features in correct order
            vision_features = []
            speech_features = []
            for feature_name in vision_features_ordered_names:
                vision_features.append(vision_features_dict[feature_name])
            for feature_name in speech_features_ordered_names:
                speech_features.append(speech_features_dict[feature_name])
            features = [torch.Tensor(vision_features).view(1, 1, -1).cuda(),
                        torch.Tensor(speech_features).view(1, 1, -1).cuda()]

            # Predict
            _, predictions = model.predict_online(*features)
            predictions = predictions.tolist()
            print(f'Predictions: {predictions}')
            predictions_buffer.append(predictions)

        # [Optional] Smooth the last N predictions in the predictions_buffer
        # ... TODO

        # Send predictions to the virtual human (Smartbody)
        # ... TODO


def preprocess_vision_features(features, previous_features, normalization_params):
    """Preprocess vision features.

    TODO

    Args:
        features (dict): Dictionary of raw input vision features from the current timestep
        previous_features (dict): Dictionary of raw input vision features from the previous timestep
        normalization_params (dict): Dictionary of normalization parameters (currently, for mean normalization)

    Returns:
        features (dict): Dictionary of preprocessed vision features for the current timestep
    """

    # Calculate first-order differences
    #   Head translations (first-order differences)
    #       'diff_ pose_Tx',
    #       'diff_ pose_Ty',
    #       'diff_ pose_Tz',
    #   Head rotations (first-order differences)
    #       'diff_ pose_Rx',
    #       'diff_ pose_Ry',
    #       'diff_ pose_Rz',
    for feature_name in vision_features_names_to_diff:
        features[f'diff_{feature_name}'] = features[feature_name] - previous_features[feature_name]

    # Mean-normalize R_x, R_y, gaze_angle_x, gaze_angle_y
    #   Head rotations (raw) as a proxy for gaze - need to be normalized (mean normalization per recording)
    #       'unorm_ pose_Rx',
    #       'unorm_ pose_Ry',
    #   Gaze angles - need to be normalized (mean normalization per recording)
    #       'unorm_ gaze_angle_x',
    #       'unorm_ gaze_angle_y',
    for feature_name in vision_features_names_to_normalize:
        features[f'unorm_{feature_name}'] = features[feature_name] - normalization_params[feature_name]

    return features


def preprocess_speech_features(features):
    """Preprocess speech features.

    Args:
        features (dict): Dictionary of raw input speech features from the current timestep

    Returns:
        features (dict): Dictionary of preprocessed speech features for the current timestep

    Currently, no preprocessing of speech features is needed.
    """

    return features


if __name__ == '__main__':
    main()
