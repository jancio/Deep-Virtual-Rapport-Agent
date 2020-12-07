### Keras code to develop the Head Gesture Detector (HGD) models, predicting head gestures (nod, shake, tilt) from video. 

	Project: Deep Virtual Rapport Agent (head gesture detector)
	Jan Ondras (jo951030@gmail.com)
	Institute for Creative Technologies, University of Southern California
	April-October 2019
------------


In this project, I developed real-time head nod, head shake, and head tilt detectors. The detectors input head rotation angles (pitch, yaw, and roll) extracted from videos using OpenFace and output binary predictions of nod, shake, and tilt head gestures on a frame-by-frame basis. The frame-by-frame predictions are made individually (binary value for each of the three head gestures) as well as using a fusion (no more than one kind of head gesture predicted for a given frame). I performed lots of experiments comparing various feature sets and window sizes, subject-dependent and subject-independent models, as well as carrying out a cross-dataset evaluation.

- Report [[PDF]](https://github.com/jancio/Deep-Virtual-Rapport-Agent/blob/master/head_gesture_detector/HeadGestureDetector_Report.pdf)
- Later deployed in *OpenSense* platform, ACM ICMI 2020, Paper [[PDF]](https://dl.acm.org/doi/abs/10.1145/3382507.3418832)

------------
Overview of the files in this folder:

	./_early_vra1_nod_only
		Scripts and checkpoints from initial experiments with nod detection using the vra1 dataset. 

	./checkpoints
		Model checkpoints (.hdf) and training history (.pkl) files. 
		The files related to training settings TS1, TS2, and TS3 can be found in the folders ./checkpoints, ./checkpoints/4comb, and ./checkpoints/final_4comb respectively. 
		The best final nod, shake, and tilt HGD models can be found in the folder ./checkpoints/final_4comb/best 
		Filenames naming convention: {dataset_name}_{head_gesture}_{window_size}ws_{number_of_features}f_{model_architecture}.{hdf5,pkl} 
		Training history files contain training parameters and train and validation evaluation metrics. 

	./evaluate_hgd.ipynb
		Test each model trained on one of the vra1, hatice2010, sewa, and nvb datasets on each of these datasets (cross-dataset evaluation). 
		So far, 4 x 3 = 12 nod only cross-dataset tests were performed. 

	./evaluate_4comb_hgd.ipynb
		Test nod, shake, and tilt detector models trained on the 4comb dataset on the test partitions of the same dataset. 

	./evaluate_final_4comb_hgd.ipynb
		Test the final nod, shake, and tilt detector models trained on the whole 4comb dataset on other datasets. 
		Namely, cross-dataset testing on the ccdb dataset. 
		This tests the nod, shake, and tilt models independently. 
		Requires the segmented ccdb datasets to be generated before (datasets_scripts/ccdb/generate_dataset.ipynb). 

	./evaluate_fused_final_4comb_hgd.ipynb
		Test the final fused 4-class (none/nod/shake/tilt) HGD model trained on the whole 4comb dataset on other datasets. 
		Namely, cross-dataset testing on the ccdb dataset. 
		Requires the final HGD (./hgd_annotate_frames.ipynb) to be run on the ccdb dataset before. 

	./hgd_annotate_frames.ipynb
		Run the developed Head Gesture Detector (HGD), trained on the whole 4comb dataset. 
		Annotates frames of given csv files of vision features (by OpenFace), using the developed Head Gesture Detector. 

	./hgd_annotate_videos.py
		Given an original video file from a dataset and vision features (csv) annotated with head gestures using the developed Head Gesture Detector, generate a new video file with color annotations of head gestures (nod, shake, tilt). 

	./multi_scale_head_gesture-master
		Head gesture detection code from the work “Recognizing Visual Signatures of Spontaneous Head Gestures” that uses the FIPCO dataset and also evaluates the developed head gesture detectors on the CCDb dataset. 
		Obtained from the author Mohit Sharma (mohits1@andrew.cmu.edu): https://github.com/mohitsharma0690/multi_scale_head_gesture 

	./Report_head_gesture_detector.odt
		Detailed description of HGD development procedures, datasets, models, experiments performed, and future work / ideas to try. 

	./show_training_history.ipynb
		Shows training history of HGD models from every training setting (TS1, TS2, TS3). 
		This was used to compare training and validation curves for various window sizes, number of features, model architectures, and evaluation metrics. (It also shows the training history when using the augmentation A1 with the training setting TS3.) 

	./test_results
		Test results saved as one .pkl file per head gesture. 
		Produced by ./evaluate_hgd.py, ./evaluate_4comb_hgd.py, and ./evaluate_final_4comb_hgd.py and saved to sub-folders test_results_{S,nonS}, test_results_4comb_{S,nonS}, and test_results_final_4comb_{S,nonS} respectively. 
		Filenames naming convention: test_results_{dataset_name}_{S,nonS}_{head_gesture}_{window_size}ws_{number_of_features}f_{model_architecture}.pkl 
		Filename infix *_S_* and *_nonS_* denotes the results obtained with and without smoothing of the sequences of the predicted binary head gesture labels respectively. 

	./train_hgd.py
		Train and validate a Head Gesture Detector model on one dataset to detect one head gesture. 
		Training setting: TS1 (subject-dependent dataset split). 
		Used with datasets (vra1, hatice2010, sewa, nvb) so that we trained and validated on one dataset and later tested on each of them (cross-dataset evaluation). 
		So far, for TS1, only the nod head gesture was used => 4 nod HGD models were trained.

	./train_4comb_hgd.py
		Train and validate a Head Gesture Detector model on the 4comb dataset to detect one head gesture. 
		Training setting: TS2 (subject-independent dataset split). 
		Uses the train and validation partitions of the 4comb dataset for training and validation. 
		The test set is untouched => can be used to evaluate the developed Head Gesture Detector on the same dataset (namely, on the 4comb test partition). 
		Used with all head gestures (nod, shake, tilt) => 3 HGD models were trained. 

	./train_final_4comb_hgd.py
		Train and validate a final Head Gesture Detector model on the whole 4comb dataset to detect one head gesture. 
		Training setting: TS3 (subject-independent dataset split). 
		The test set is also used for training (train = train + test, val = val) => the developed Head Gesture Detector can be evaluated on other datasets (e.g., cross-dataset testing on ccdb) and used for general prediction of head gestures. 
		Used with all head gestures (nod, shake, tilt) => 3 HGD models were trained. 

	./utils.py
		Helper functions for HGD training/testing: 
		 - Evaluate custom metrics. 
		 - Plot training and validation curves for various evaluation metrics. 
		 - Plot test results by subjects. 
		 - Save and load training history. 
