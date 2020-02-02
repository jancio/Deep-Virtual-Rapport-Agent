#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# [Currently, just a skeleton code.]
# Record normalization parameters (means) for online prediction.
#     Namely:
#         2 mean head rotations (from ‘ pose_Rx’ and ‘ pose_Ry’ by OpenFace) 
#         2 mean gaze angles (from ‘ gaze_angle_x’ and ‘ gaze_angle_y’ by OpenFace) 
#     
#     During the calibration recording, the speaker should look directly at the virtual human. 
# 
#     Saves the parameters as .npz file in the ./normalization_params folder.
#     Uses Openface to determine normalization parameters for vision features.
#     No normalization is needed for speech features by Opensmile.
#######################################################################################################################

import os
import time
import argparse
import numpy as np
from collections import defaultdict

from colors import ConsoleColors
from constants import vision_features_names_to_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording_time', default=30., type=float,
                        help='recording time (seconds) over which the normalization parameters will be calculated')
    parser.add_argument('--normalization_params_dir',
                        default='/home/ICT2000/jondras/deep-virtual-rapport-agent/rapport_model/normalization_params',
                        help='path to save normalization parameters for online prediction')
    parser.add_argument('--normalization_params_file_suffix', default='',
                        help='suffix of the file with normalization parameters')
    opt = parser.parse_args()

    # Accumulate vision features from the recording
    vision_features = defaultdict(list)

    # Main loop
    start_time = time.time()
    while (time.time() - start_time) <= opt.recording_time:
        # Get vision features from OpenFace
        # ... TODO
        # vision_feature_vector =

        for feature_name in vision_features_names_to_normalize:
            vision_features[feature_name].append(vision_feature_vector[feature_name])

    # Save normalization parameters: means
    normalization_params = dict()
    for feature_name in vision_features_names_to_normalize:
        normalization_params[feature_name] = np.mean(vision_features[feature_name])

    normalization_params_file = os.path.join(opt.normalization_params_dir,
                                        f'{int(time.time())}_norm_params_{opt.normalization_params_file_suffix}.npz')
    np.savez(normalization_params_file, **normalization_params)
    print(f'Normalization parameters\n\t {ConsoleColors.CC_YELLOW}{normalization_params}{ConsoleColors.CC_END}\n'
          f'were saved to\n\t {normalization_params_file}')


if __name__ == '__main__':
    main()
