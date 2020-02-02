### Scripts to preprocess the hatice2010 dataset

	Project: Deep Virtual Rapport Agent
	Jan Ondras (jo951030@gmail.com)
	Institute for Creative Technologies, University of Southern California
	April-October 2019
------------


Preprocessing pipeline: 

	annotate_features.ipynb --> generate_dataset.ipynb


Overview of the files in this folder:
	
	./annotate_features.ipynb
		Annotate OpenFace features from the hatice2010 dataset with ground-true head gestures (nod and shake).

	./generate_dataset.ipynb
		Generate segmented/sequenced dataset from hatice2010 dataset. 
		For head gestures nod and shake. 
		The generated dataset was used for the development of the Head Gesture Detector.
