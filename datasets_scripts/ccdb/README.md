### Scripts to preprocess the ccdb dataset

	Project: Deep Virtual Rapport Agent
	Jan Ondras (jo951030@gmail.com)
	Institute for Creative Technologies, University of Southern California
	April-October 2019
------------


Preprocessing pipeline: 

	extract_features.sh --> annotate_features.ipynb --> generate_dataset.ipynb


Overview of the files in this folder:
	
	./access
		Login details to access the ccdb dataset and EULA.

	./annotate_features.ipynb
		Annotate OpenFace features from the ccdb dataset with ground-true head gestures (nod, shake, and tilt). 
		So far only for the basic set of the first 8 sessions. 
		Note: the script to annotate features using the developed Head Gesture Detector (and not the ground-truths) is deep-virtual-rapport-agent/head_gesture_detector/hgd_annotate_frames.ipynb and the corresponding annotated features are in dvra_datasets/ccdb/hgd_annotated_features

	./extract_features.sh
		Extract vision features from the ccdb dataset using OpenFace.

	./feature_extraction_time_log.txt
		Log of (filepath, filename, and OpenFace vision features extraction time (in seconds)) for each video in the ccdb dataset.

	./generate_dataset.ipynb
		Generate segmented/sequenced test dataset from ccdb dataset. 
		For head gestures nod, shake, and tilt. 
		So far only for the basic set of the first 8 sessions. 
		The generated dataset was used for cross-dataset testing of the developed Head Gesture Detector. 
		In future, the ccdb dataset can be included in the training data, extending the 4comb dataset to 5comb.
