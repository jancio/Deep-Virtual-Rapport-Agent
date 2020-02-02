### Scripts to preprocess the sewa dataset

	Project: Deep Virtual Rapport Agent
	Jan Ondras (jo951030@gmail.com)
	Institute for Creative Technologies, University of Southern California
	April-October 2019
------------


Preprocessing pipeline: 

	extract_features.sh --> annotate_features.ipynb --> generate_dataset.ipynb



Overview of the files in this folder:
	
	./annotate_features.ipynb
		Annotate OpenFace features from the sewa dataset with ground-true head gestures (nod and shake).

	./extract_features.sh
		Extract vision features from the sewa dataset using OpenFace.

	./feature_extraction_time_log.txt
		Log of (filepath, filename, and OpenFace vision features extraction time (in seconds)) for each video in the sewa dataset.

	./generate_dataset.ipynb
		Generate segmented/sequenced dataset from sewa dataset. 
		For head gestures nod and shake. 
		The generated dataset was used for the development of the Head Gesture Detector.
