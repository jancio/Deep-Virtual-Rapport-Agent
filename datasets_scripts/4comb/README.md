### Scripts to preprocess the 4comb dataset (combination of vra1, sewa, nvb, and hatice2010)

	Project: Deep Virtual Rapport Agent
	Jan Ondras (jo951030@gmail.com)
	Institute for Creative Technologies, University of Southern California
	April-October 2019
------------


Overview of the files in this folder:
	
	./generate_4comb_dataset.ipynb
		Generate segmented/sequenced dataset from vra1, sewa, hatice2010, and nvb datasets. 
		For all head gestures (nod, shake, tilt). 
		The annotate_features.ipynb scripts of these datasets need to be run first! 
		The generated dataset was used for the development of the Head Gesture Detector.
		In future, the ccdb dataset can also be included in the training data, extending the 4comb dataset to 5comb.
		Print outputs from this script were also saved in deep-virtual-rapport-agent/notes/results/log_generate_4comb_dataset.docx
