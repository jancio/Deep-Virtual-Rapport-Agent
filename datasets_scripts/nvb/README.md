### Scripts to preprocess the nvb dataset

	Project: Deep Virtual Rapport Agent
	Jan Ondras (jo951030@gmail.com)
	Institute for Creative Technologies, University of Southern California
	April-October 2019
------------


Preprocessing pipeline: 

	unify_filenames.ipynb --> extract_features.sh --> annotate_features.ipynb --> generate_dataset.ipynb



Overview of the files in this folder:
	
	./annotate_features.ipynb
		Annotate OpenFace features from the nvb dataset with ground-true head gestures (nod, shake, and tilt). 
		Note: Make sure the unify_filenames.ipynb script was run!


	./extract_features.sh
		Extract vision features from the nvb dataset using OpenFace.

	./extract_frame_to_check_subjects.sh
		Extract 1 frame from each video to check whether all subjects differ (i.e., to manually verify that each video contains a different subject).

	./feature_extraction_time_log.txt
		Log of (filepath, filename, and OpenFace vision features extraction time (in seconds)) for each video in the nvb dataset.

	./generate_dataset.ipynb
		Generate segmented/sequenced dataset from nvb dataset. 
		For head gestures nod, shake, and tilt. 
		The generated dataset was used for the development of the Head Gesture Detector.


	./unify_filenames.ipynb
		Rename EAF annotation filenames in the nvb dataset to common format {sessid}.eaf. 
		Run only once and before running the annotate_features.ipynb script!
