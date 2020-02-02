### Scripts to preprocess the vra1 dataset

	Project: Deep Virtual Rapport Agent
	Jan Ondras (jo951030@gmail.com)
	Institute for Creative Technologies, University of Southern California
	April-October 2019
------------


Preprocessing pipeline: 

	aggregate_all_*.sh --> unify_filenames.ipynb --> listener_extract_features.sh --> annotate_features.ipynb --> generate_dataset.ipynb


Overview of the files in this folder:
	
	./aggregate_all_annotations.sh
		Aggregate all annotations of listener nods into one folder. 

	./aggregate_all_videos.sh
		Aggregate all listener videos into one folder. 

	./annotate_features.ipynb
		Annotate listener OpenFace features from the vra1 dataset with ground-true head gestures (nod). 
		Note: Make sure the unify_filenames.ipynb script was run! 

	./generate_dataset.ipynb
		Generate segmented/sequenced dataset from vra1 dataset. 
		For nod head gesture. 
		When aligning vision features and head gesture annotations, the offsets from offsetListenerNods.txt are used (to ignore the beginnings of recordings prior to the beep). 
		The generated dataset was used for the development of the Head Gesture Detector.

	./listener_extract_features.sh
		Extract listener vision features from the vra1 dataset using OpenFace. 

	./listener_feature_extraction_time_log.txt
		Log of (filename, and listener's vision features extraction time by OpenFace (in seconds)) for each video in the vra1 dataset. 

	./offsetListenerNods.txt
		Time offsets: one for each annotation file of listener nods. 
		Used to generate dataset for the development of the Head Gesture Detector. 
		Will also need to be used to generate dataset for the development of the Rapport Model. 

	./offsetSpeakerAudio.txt
		Time offsets: one for each audio file of a speaker. 
		Will need to be used to generate dataset for the development of the Rapport Model. 

	./unify_filenames.ipynb
		Rename listener EAF annotation filenames and listener video filenames in the vra1 dataset to unified formats: SES{id}.eaf and SES{id}.mp4 
		Run only once and after aggregate_all_*.sh scripts and manual filtering! 
