#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Examples and experiments using the scripts (train/test/final_train/record_normalization_params/predict_online.py).
#######################################################################################################################

# Example: train: vision modality only, unimodal base classifier, one target label only (nod)
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v3

# Example: train: multimodal, multimodal TCN classifier, 3 target labels (nod, shake, tilt)
python train.py --modality multimodal --model_type multimodal-tcn-classifier --labels_names nod shake tilt --logger_dir_suffix NST --gpu_id 0 --vision_rnn_layer_dim 32 --speech_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v3

# Example: testing after train
python train.py --modality multimodal --model_type multimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --vision_rnn_layer_dim 32 --speech_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v3
python test.py --logger_path /home/ICT2000/jondras/deep-virtual-rapport-agent/rapport_model/logs/1569294550_multimodal_multimodal-base-classifier_nod --dataset_version v3

# Example: record normalization parameters
python record_normalization_params.py --recording_time 15

# Example: final train and online predition
python final_train.py --modality multimodal --model_type multimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --vision_rnn_layer_dim 32 --speech_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v3
python predict_online.py --logger_path /home/ICT2000/jondras/deep-virtual-rapport-agent/rapport_model/logs/1569294550_multimodal_multimodal-base-classifier_nod

#######################################################################################################################
# Experiments (everything below)
#######################################################################################################################
# Experiments #1

# SPEECH - SINGLE LABEL
python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --rnn_layer_dim 64 --epochs_num 100
python train.py --modality speech --model_type unimodal-base-classifier --labels_names shake --logger_dir_suffix shake --gpu_id 0 --rnn_layer_dim 64 --epochs_num 100
python train.py --modality speech --model_type unimodal-base-classifier --labels_names tilt --logger_dir_suffix tilt --gpu_id 0 --rnn_layer_dim 64 --epochs_num 100
python train.py --modality speech --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix smile --gpu_id 1 --rnn_layer_dim 64 --epochs_num 100
python train.py --modality speech --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix gaze_away --gpu_id 1 --rnn_layer_dim 64 --epochs_num 100
python train.py --modality speech --model_type unimodal-base-classifier --labels_names voice_active --logger_dir_suffix voice_active --gpu_id 1 --rnn_layer_dim 64 --epochs_num 100
python train.py --modality speech --model_type unimodal-base-classifier --labels_names take_turn --logger_dir_suffix take_turn --gpu_id 1 --rnn_layer_dim 64 --epochs_num 100

# VISION - SINGLE LABEL
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --rnn_layer_dim 32 --epochs_num 100
python train.py --modality vision --model_type unimodal-base-classifier --labels_names shake --logger_dir_suffix shake --gpu_id 0 --rnn_layer_dim 32 --epochs_num 100
python train.py --modality vision --model_type unimodal-base-classifier --labels_names tilt --logger_dir_suffix tilt --gpu_id 0 --rnn_layer_dim 32 --epochs_num 100
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix smile --gpu_id 1 --rnn_layer_dim 32 --epochs_num 100
python train.py --modality vision --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix gaze_away --gpu_id 1 --rnn_layer_dim 32 --epochs_num 100
python train.py --modality vision --model_type unimodal-base-classifier --labels_names voice_active --logger_dir_suffix voice_active --gpu_id 1 --rnn_layer_dim 32 --epochs_num 100
python train.py --modality vision --model_type unimodal-base-classifier --labels_names take_turn --logger_dir_suffix take_turn --gpu_id 1 --rnn_layer_dim 32 --epochs_num 100

# MULTIMODAL - SINGLE LABEL
python train.py --modality multimodal --model_type multimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --vision_rnn_layer_dim 32 --speech_rnn_layer_dim 64 --epochs_num 100
python train.py --modality multimodal --model_type multimodal-base-classifier --labels_names shake --logger_dir_suffix shake --gpu_id 0 --vision_rnn_layer_dim 32 --speech_rnn_layer_dim 64 --epochs_num 100
python train.py --modality multimodal --model_type multimodal-base-classifier --labels_names tilt --logger_dir_suffix tilt --gpu_id 0 --vision_rnn_layer_dim 32 --speech_rnn_layer_dim 64 --epochs_num 100
python train.py --modality multimodal --model_type multimodal-base-classifier --labels_names smile --logger_dir_suffix smile --gpu_id 1 --vision_rnn_layer_dim 32 --speech_rnn_layer_dim 64 --epochs_num 100
python train.py --modality multimodal --model_type multimodal-base-classifier --labels_names gaze_away --logger_dir_suffix gaze_away --gpu_id 1 --vision_rnn_layer_dim 32 --speech_rnn_layer_dim 64 --epochs_num 100
python train.py --modality multimodal --model_type multimodal-base-classifier --labels_names voice_active --logger_dir_suffix voice_active --gpu_id 1 --vision_rnn_layer_dim 32 --speech_rnn_layer_dim 64 --epochs_num 100
python train.py --modality multimodal --model_type multimodal-base-classifier --labels_names take_turn --logger_dir_suffix take_turn --gpu_id 1 --vision_rnn_layer_dim 32 --speech_rnn_layer_dim 64 --epochs_num 100


#######################################################################################################################
# Experiments #2 (29 August)

# NOD - vision

python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --vision_rnn_layer_dim 16 --epochs_num 100 --batch_size 128
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --vision_rnn_layer_dim 32 --epochs_num 100 --batch_size 128 --monitored_metric bacc

# with just one fully-connected layer after RNN, no dropout, ...
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --vision_rnn_layer_dim 32 --epochs_num 100 --batch_size 128 --monitored_metric bacc

# increase # GRU units with just one fully-connected layer after RNN, no dropout, ...
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc

# increase # GRU units with original 2 fully-connected layers after RNN, and dropout, ...
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix nod --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc

# increase window size to 16 frames (> 3 sec)
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16_nod --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 16

# decrease window size to 4 frames (< 1 sec)
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 4_nod --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 4

# increase # GRU units to 128
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16ws_128u_nod --gpu_id 0 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 16

# with 64 GRU units add 3rd FC layer
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16ws_64u_3fc_nod --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 3

# with 128 GRU units add 3rd FC layer
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16ws_128u_3fc_nod --gpu_id 0 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 3

# 32 window size
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_128u_3fc_nod --gpu_id 3 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 3

#######################################################################################################################
# Experiments #3 (30 August)

# NOD - vision

# v1 dataset: 128 GRU, window sizes: 8, 16, 32; 3 FC

python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 8ws_128u_3fc_nod --gpu_id 2 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 3 --dataset_version v1
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16ws_128u_3fc_nod --gpu_id 2 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 3 --dataset_version v1
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_128u_3fc_nod --gpu_id 2 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 3 --dataset_version v1

# same but for v0 dataset
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 8ws_128u_3fc_nod_v0 --gpu_id 1 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 3 --dataset_version v0
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16ws_128u_3fc_nod_v0 --gpu_id 1 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 3 --dataset_version v0
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_128u_3fc_nod_v0 --gpu_id 1 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 3 --dataset_version v0


# v0 vs v1:

# NOD - audio
python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 8ws_128u_3fc_nod_v0 --gpu_id 0 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 3 --dataset_version v0
python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16ws_128u_3fc_nod_v0 --gpu_id 0 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 3 --dataset_version v0
python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_128u_3fc_nod_v0 --gpu_id 0 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 3 --dataset_version v0

python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 8ws_128u_3fc_nod_v1 --gpu_id 0 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 3 --dataset_version v1
python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16ws_128u_3fc_nod_v1 --gpu_id 0 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 3 --dataset_version v1
python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_128u_3fc_nod_v1 --gpu_id 0 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 3 --dataset_version v1

# GAZE - audio
python train.py --modality speech --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 8ws_128u_3fc_gaze_v0 --gpu_id 2 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 3 --dataset_version v0
python train.py --modality speech --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 16ws_128u_3fc_gaze_v0 --gpu_id 2 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 3 --dataset_version v0
python train.py --modality speech --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 32ws_128u_3fc_gaze_v0 --gpu_id 2 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 3 --dataset_version v0

python train.py --modality speech --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 8ws_128u_3fc_gaze_v1 --gpu_id 2 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 3 --dataset_version v1
python train.py --modality speech --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 16ws_128u_3fc_gaze_v1 --gpu_id 2 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 3 --dataset_version v1
python train.py --modality speech --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 32ws_128u_3fc_gaze_v1 --gpu_id 2 --speech_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 3 --dataset_version v1

# SMILE - vision
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 8ws_128u_3fc_smile_v0 --gpu_id 1 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 3 --dataset_version v0
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 16ws_128u_3fc_smile_v0 --gpu_id 1 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 3 --dataset_version v0
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 32ws_128u_3fc_smile_v0 --gpu_id 1 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 3 --dataset_version v0

python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 8ws_128u_3fc_smile_v1 --gpu_id 1 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 3 --dataset_version v1
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 16ws_128u_3fc_smile_v1 --gpu_id 1 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 3 --dataset_version v1
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 32ws_128u_3fc_smile_v1 --gpu_id 1 --vision_rnn_layer_dim 128 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 3 --dataset_version v1

#######################################################################################################################
# Experiments #4 (3 September)

# v2 vs v1:

# NOD - vision
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 8ws_64u_2fc_nod_v2 --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16ws_64u_2fc_nod_v2 --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_64u_2fc_nod_v2 --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2

python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 8ws_64u_2fc_nod_v1 --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 2 --dataset_version v1
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16ws_64u_2fc_nod_v1 --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v1
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_64u_2fc_nod_v1 --gpu_id 0 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v1

# SMILE - vision
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 8ws_64u_2fc_smile_v2 --gpu_id 1 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 16ws_64u_2fc_smile_v2 --gpu_id 1 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 32ws_64u_2fc_smile_v2 --gpu_id 1 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2

python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 8ws_64u_2fc_smile_v1 --gpu_id 1 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 2 --dataset_version v1
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 16ws_64u_2fc_smile_v1 --gpu_id 1 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v1
python train.py --modality vision --model_type unimodal-base-classifier --labels_names smile --logger_dir_suffix 32ws_64u_2fc_smile_v1 --gpu_id 1 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v1

# GAZE - vision
python train.py --modality vision --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 8ws_64u_2fc_gaze_v2 --gpu_id 2 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 16ws_64u_2fc_gaze_v2 --gpu_id 2 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 32ws_64u_2fc_gaze_v2 --gpu_id 2 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2

python train.py --modality vision --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 8ws_64u_2fc_gaze_v1 --gpu_id 2 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 2 --dataset_version v1
python train.py --modality vision --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 16ws_64u_2fc_gaze_v1 --gpu_id 2 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v1
python train.py --modality vision --model_type unimodal-base-classifier --labels_names gaze_away --logger_dir_suffix 32ws_64u_2fc_gaze_v1 --gpu_id 2 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v1

#######################################################################################################################
# Experiments #5 (4 September)

# smaller network sizes for nod-vision 
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_32u_2fc_nod_v2 --gpu_id 1 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_16u_2fc_nod_v2 --gpu_id 1 --vision_rnn_layer_dim 16 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2

#######################################################################################################################
# Experiments #6 (5 September)

# TCN vision nod
python train.py --modality vision --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 32ws_32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 16ws_32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 8ws_32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 2 --dataset_version v2

# TCN vs base: nod audio
python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_32u_2fc_nod_v2_TCN --gpu_id 2 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2
python train.py --modality speech --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 32ws_32u_2fc_nod_v2_TCN --gpu_id 2 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2

python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 16ws_32u_2fc_nod_v2_TCN --gpu_id 2 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v2
python train.py --modality speech --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 16ws_32u_2fc_nod_v2_TCN --gpu_id 2 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v2

python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 8ws_32u_2fc_nod_v2_TCN --gpu_id 2 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 2 --dataset_version v2
python train.py --modality speech --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 8ws_32u_2fc_nod_v2_TCN --gpu_id 2 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 8 --fc_layer_num 2 --dataset_version v2

#######################################################################################################################
# Experiments #7 (6 September)

# vision

# 2 TCN layers
python train.py --modality vision --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 32ws_32u32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --rnn_layer_num 2 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 16ws_32u32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --rnn_layer_num 2 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v2

# 3 TCN layers
python train.py --modality vision --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 32ws_32u32u32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --rnn_layer_num 3 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2
python train.py --modality vision --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 16ws_32u32u32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --rnn_layer_num 3 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v2

# speech

# 2 TCN layers
python train.py --modality speech --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 32ws_32u32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --rnn_layer_num 2 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2
python train.py --modality speech --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 16ws_32u32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --rnn_layer_num 2 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v2

# 3 TCN layers
python train.py --modality speech --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 32ws_32u32u32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --rnn_layer_num 3 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v2
python train.py --modality speech --model_type unimodal-tcn-classifier --labels_names nod --logger_dir_suffix 16ws_32u32u32u_2fc_nod_v2_TCN --gpu_id 1 --vision_rnn_layer_dim 32 --rnn_layer_num 3 --epochs_num 100 --monitored_metric bacc --sequence_length 16 --fc_layer_num 2 --dataset_version v2

#######################################################################################################################
# Experiments #8 (23 September)

# v3 dataset: 32ws (at 30Hz => about 1 sec)

# vision-nod
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_16u_2fc_nod_v3 --gpu_id 2 --vision_rnn_layer_dim 16 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v3
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_32u_2fc_nod_v3 --gpu_id 2 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v3
python train.py --modality vision --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_64u_2fc_nod_v3 --gpu_id 2 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v3

# speech-nod
python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_16u_2fc_nod_v3 --gpu_id 2 --vision_rnn_layer_dim 16 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v3
python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_32u_2fc_nod_v3 --gpu_id 2 --vision_rnn_layer_dim 32 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v3
python train.py --modality speech --model_type unimodal-base-classifier --labels_names nod --logger_dir_suffix 32ws_64u_2fc_nod_v3 --gpu_id 2 --vision_rnn_layer_dim 64 --epochs_num 100 --monitored_metric bacc --sequence_length 32 --fc_layer_num 2 --dataset_version v3
