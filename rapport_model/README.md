### PyTorch code to develop rapport models, predicting listener's behavior based on the speaker's audio and video

	Project: Deep Virtual Rapport Agent (rapport model)
	Jan Ondras (jo951030@gmail.com)
	Institute for Creative Technologies, University of Southern California
	April-October 2019
------------


The aim of this project was to develop a deep learning model to generate a virtual agent listener’s real-time rapport behavior, given a human speaker’s audio-visual input. Using a multimodal dataset of dyadic interactions, I trained several machine learning models to predict a virtual agent’s head gestures, smile, gaze, voice activity, and turn-taking. To annotate the training data with head gestures (nod, shake, and tilt), I used the previously developed [Head Gesture Detector](https://github.com/jancio/Deep-Virtual-Rapport-Agent/tree/master/head_gesture_detector). This project also involved voice activity detection to determine time intervals when a human speaker is speaking and thus filter out irrelevant data. For this sub-task, I experimented with various toolkits such as OpenSMILE, IBM Watson STT, Google ASR STT, and WebRTC VAD.

- Report [[PDF]](https://github.com/jancio/Deep-Virtual-Rapport-Agent/blob/master/rapport_model/RapportModel_Report.pdf)


------------
[The code was originally adapted from Kalin Stefanov's PyTorch codebase (https://github.com/intelligent-human-perception-laboratory/pytorch-codebase/tree/kalins-perception).]

Overview of the files in this folder:

	./data.py
		PyTorch dataset classes and dataset loaders for rapport model training/testing.
		Both unimodal and multimodal variants.

	./model.py
		PyTorch rapport models.
		Both unimodal and multimodal variants.
		Currently implemented:
			GRU-based base classifier,
			Temporal Convolutional Network (TCN) classifier

	./train.py
		Train and validate rapport models.
		Both subject-dependent and subject-independent cross-validation variants.

	./test.py
		Test rapport models on the hold-out test sets from all folds (from either subject-dependent or subject-independent cross-validation).
		Logs from testing are saved in the same logging directory as provided in the logger_path argument.

	./final_train.py
		Final training of rapport models on the whole dataset, for online prediction.
		No hold-out test set created.
		No cross-validation.
		Single-split validation is either subject-dependent or subject-independent.
		Otherwise, based on ./train.py.

	./record_normalization_params.py
		[Currently, just a skeleton code.]
		Record normalization parameters (means) for online prediction.
		Saves the parameters as .npz file in the ./normalization_params folder.
		Uses OpenFace to determine normalization parameters for vision features.
		No normalization is needed for speech features by OpenSMILE.

	./predict_online.py
		[Currently, just a skeleton code.]
		Online prediction using rapport models.
		For now, to simulate the online prediction, there is an extra code within the "ONLY FOR SIMULATION OF ONLINE PREDICTION" blocks.
		Logs from online prediction are saved in the same logging directory as provided in the logger_path argument.

	./colors.py
		Definition of colors for console pretty-printing.

	./constants.py
		Definition of constants such as labels/class names and features names.

	./utils.py
		Helper functions for training/testing.

	./logger.py
		Dual logger class to print outputs to both the terminal and a log file.

	./run.sh
		Examples of commandline calls of the above scripts (train/test/final_train/record_normalization_params/predict_online).

	./Report_rapport_model.odt
		Detailed description of development procedures, the dataset, models, experiments performed, and future work / ideas to try.

	./logs
		Directory of logging directories. 
		One logging directory is created for each run.
		The format of logging directory names is:
			{timestamp}_{modality}_{model_type}_{logger_dir_suffix}
		Logging directory may contain:
			'fold_*' directories with (last and best) models for each cross-validation fold
			'final' directory with the final model trained on the whole dataset
			'{train/test/final_train/predict_online}.log' files with stdout/stderr outputs from train/test/final_train/predict_online
			tensorboard binary file
		To view the runs/logs using tensorboard:
			Install tensorboard from: https://pytorch.org/docs/stable/tensorboard.html
			Run the command: tensorboard --logdir logs
			In your browser, go to: http://cronos:6006/
			Filter plots by tag names, e.g.:
				fold_01/train/nod/bacc		shows the training balanced accuracy of nod class/label for fold 1
				fold_.*/val/nod/loss		shows the validation loss of nod class/label for all folds

	./normalization_params
		Directory of files with normalization parameters for online prediction.
		The format of these filenames is:
			{timestamp}_norm_params_{normalization_params_file_suffix}.npz

	./TCN
		Directory with the Temporal Convolutional Network (TCN) model. 
		Cloned from https://github.com/locuslab/TCN
