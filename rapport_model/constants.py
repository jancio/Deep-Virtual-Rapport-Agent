#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Constants for training/testing the rapport model.
#######################################################################################################################


# Set of all names of predicted labels/classes
all_labels_names_set = set([
    'nod',
    'shake',
    'tilt',
    'smile',
    'gaze_away',
    'voice_active',
    'take_turn'
])

# Names of metrics subject to minimization (all other metrics will be implicitly maximized)
minimize_metrics = ['loss']

# Names of vision features (ordered as required for rapport models)
vision_features_names = dict()
vision_features_names['v3'] = [
    # Head translations (first-order differences)
    'diff_ pose_Tx',
    'diff_ pose_Ty',
    'diff_ pose_Tz',
    # Head rotations (first-order differences)
    'diff_ pose_Rx',
    'diff_ pose_Ry',
    'diff_ pose_Rz',
    # Head rotations (raw) as a proxy for gaze - need to be normalized (mean normalization per recording)
    'unorm_ pose_Rx',
    'unorm_ pose_Ry',
    # Gaze angles - need to be normalized (mean normalization per recording)
    'unorm_ gaze_angle_x',
    'unorm_ gaze_angle_y',
    # Smile (binary)
    #     'smile'
    # Smile (raw AU intensities)
    #     ' AU06_r',
    #     ' AU12_r'
    # All AU intensities
    ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
    ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r'
]

# Names of vision features that need to be mean-normalized
vision_features_names_to_normalize = [
    # Head rotations (raw) as a proxy for gaze
    ' pose_Rx',
    ' pose_Ry',
    # Gaze angles
    ' gaze_angle_x',
    ' gaze_angle_y'
]

# Names of vision features whose first-order differences need to be calculated during online prediction
vision_features_names_to_diff = [
    # Head translations (first-order differences)
    ' pose_Tx',
    ' pose_Ty',
    ' pose_Tz',
    # Head rotations (first-order differences)
    ' pose_Rx',
    ' pose_Ry',
    ' pose_Rz'
]

# Names of audio/speech features (ordered as required for rapport models)
speech_features_names = dict()
speech_features_names['v3'] = {
    'emobase': [
        # Speaking time (in seconds)
        'speaking_time',
        # 52 emobase audio features
        'pcm_intensity_sma', 'pcm_loudness_sma', 'mfcc_sma[1]',
        'mfcc_sma[2]', 'mfcc_sma[3]', 'mfcc_sma[4]', 'mfcc_sma[5]',
        'mfcc_sma[6]', 'mfcc_sma[7]', 'mfcc_sma[8]', 'mfcc_sma[9]',
        'mfcc_sma[10]', 'mfcc_sma[11]', 'mfcc_sma[12]', 'lspFreq_sma[0]',
        'lspFreq_sma[1]', 'lspFreq_sma[2]', 'lspFreq_sma[3]', 'lspFreq_sma[4]',
        'lspFreq_sma[5]', 'lspFreq_sma[6]', 'lspFreq_sma[7]', 'pcm_zcr_sma',
        'voiceProb_sma', 'F0_sma', 'F0env_sma', 'pcm_intensity_sma_de',
        'pcm_loudness_sma_de', 'mfcc_sma_de[1]', 'mfcc_sma_de[2]',
        'mfcc_sma_de[3]', 'mfcc_sma_de[4]', 'mfcc_sma_de[5]', 'mfcc_sma_de[6]',
        'mfcc_sma_de[7]', 'mfcc_sma_de[8]', 'mfcc_sma_de[9]', 'mfcc_sma_de[10]',
        'mfcc_sma_de[11]', 'mfcc_sma_de[12]', 'lspFreq_sma_de[0]',
        'lspFreq_sma_de[1]', 'lspFreq_sma_de[2]', 'lspFreq_sma_de[3]',
        'lspFreq_sma_de[4]', 'lspFreq_sma_de[5]', 'lspFreq_sma_de[6]',
        'lspFreq_sma_de[7]', 'pcm_zcr_sma_de', 'voiceProb_sma_de', 'F0_sma_de',
        'F0env_sma_de'
    ],
    'mfcc': [
        # Speaking time (in seconds)
        'speaking_time',
        # 57 mfcc (extended) features
        'voiceProb', 'F0', 'F0env', 'pcm_intensity', 'pcm_loudness', 'pcm_LOGenergy',
        'pcm_fftMag_mfcc[0]', 'pcm_fftMag_mfcc[1]', 'pcm_fftMag_mfcc[2]', 'pcm_fftMag_mfcc[3]', 'pcm_fftMag_mfcc[4]',
        'pcm_fftMag_mfcc[5]', 'pcm_fftMag_mfcc[6]', 'pcm_fftMag_mfcc[7]', 'pcm_fftMag_mfcc[8]', 'pcm_fftMag_mfcc[9]',
        'pcm_fftMag_mfcc[10]', 'pcm_fftMag_mfcc[11]', 'pcm_fftMag_mfcc[12]',
        'voiceProb_de', 'F0_de', 'F0env_de', 'pcm_intensity_de', 'pcm_loudness_de', 'pcm_LOGenergy_de',
        'pcm_fftMag_mfcc_de[0]', 'pcm_fftMag_mfcc_de[1]', 'pcm_fftMag_mfcc_de[2]', 'pcm_fftMag_mfcc_de[3]',
        'pcm_fftMag_mfcc_de[4]',
        'pcm_fftMag_mfcc_de[5]', 'pcm_fftMag_mfcc_de[6]', 'pcm_fftMag_mfcc_de[7]', 'pcm_fftMag_mfcc_de[8]',
        'pcm_fftMag_mfcc_de[9]',
        'pcm_fftMag_mfcc_de[10]', 'pcm_fftMag_mfcc_de[11]', 'pcm_fftMag_mfcc_de[12]',
        'voiceProb_de_de', 'F0_de_de', 'F0env_de_de', 'pcm_intensity_de_de', 'pcm_loudness_de_de',
        'pcm_LOGenergy_de_de',
        'pcm_fftMag_mfcc_de_de[0]', 'pcm_fftMag_mfcc_de_de[1]', 'pcm_fftMag_mfcc_de_de[2]', 'pcm_fftMag_mfcc_de_de[3]',
        'pcm_fftMag_mfcc_de_de[4]',
        'pcm_fftMag_mfcc_de_de[5]', 'pcm_fftMag_mfcc_de_de[6]', 'pcm_fftMag_mfcc_de_de[7]', 'pcm_fftMag_mfcc_de_de[8]',
        'pcm_fftMag_mfcc_de_de[9]',
        'pcm_fftMag_mfcc_de_de[10]', 'pcm_fftMag_mfcc_de_de[11]', 'pcm_fftMag_mfcc_de_de[12]'
    ]
}
