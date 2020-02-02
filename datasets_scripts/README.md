### Scripts to preprocess datasets (vra1, sewa, nvb, hatice2010, 4comb, ccdb, mimicry)

	Project: Deep Virtual Rapport Agent
	Jan Ondras (jo951030@gmail.com)
	Institute for Creative Technologies, University of Southern California
	April-October 2019
------------


The original and preprocessed data can be found in the folder dvra_datasets in JankosPackage. The directory structure of the current folder (datasets_scripts) follows the structure of the dvra_datasets folder. 


Overview of the files in this folder:
	
	./4comb
		Scripts to preprocess the 4comb dataset (combination of vra1, sewa, nvb, and hatice2010). 
		Used for training the final Head Gesture Detector.

	./ccdb
		Scripts to preprocess the ccdb dataset. 
		Used for cross-dataset evaluation of the final Head Gesture Detector.

	./hatice2010
		Scripts to preprocess the hatice2010 dataset. 
		Used for the development of the Head Gesture Detector.

	./mimicry
		Scripts to preprocess the mimicry dataset. 
		Used for the development of the Rapport Model. 

	./nvb
		Scripts to preprocess the nvb dataset. 
		Used for the development of the Head Gesture Detector. 

	./sewa
		Scripts to preprocess the sewa dataset. 
		Used for the development of the Head Gesture Detector. 

	./vra1
		Scripts to preprocess the vra1 dataset. 
		Used for initial experiments with nod detection and also for the development of the Head Gesture Detector.

	./copy_datasets_scripts.sh
		[no longer needed] Used to copy scripts from directories with mixed data (actual datasets/features mixed with scripts to preprocess/analyze them).
