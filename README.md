### Deep Virtual Rapport Agent

	Project: Deep Virtual Rapport Agent
	Jan Ondras (jo951030@gmail.com)
	Institute for Creative Technologies, University of Southern California
	April-October 2019
------------


The aim of this project was to develop a deep learning model to generate a virtual agent listener’s real-time rapport behavior, given a human speaker’s audio-visual input. Using a multimodal dataset of dyadic interactions, I trained several machine learning models to predict a virtual agent’s head gestures, smile, gaze, voice activity, and turn-taking. To annotate the training data with head gestures (nod, shake, and tilt), I used the previously developed [Head Gesture Detector](https://github.com/jancio/Deep-Virtual-Rapport-Agent/tree/master/head_gesture_detector). This project also involved voice activity detection to determine time intervals when a human speaker is speaking and thus filter out irrelevant data. For this sub-task, I experimented with various toolkits such as OpenSMILE, IBM Watson STT, Google ASR STT, and WebRTC VAD.

- Report [[PDF]](https://github.com/jancio/Deep-Virtual-Rapport-Agent/blob/master/rapport_model/RapportModel_Report.pdf)


------------
Overview of the files in this folder:
	
	./data_analysis
		Scripts to analyze and calculate statistics of datasets (vra1, sewa, nvb, hatice2010, ccdb, mimicry).

	./datasets_scripts
		Scripts to preprocess datasets (vra1, sewa, nvb, hatice2010, 4comb, ccdb, mimicry).

	./head_gesture_detector
		The Head Gesture Detector code. 
		Includes training, evaluation, model checkpoints, and prediction.

	./notes
		Notes related to the Deep Virtual Rapport Agent project.
		Includes: notes on related work, datasets, models, OpenFace, SmartBody, and BML; results from head gesture detection; and my cheatsheet. 

	./rapport_model
		The Rapport Model code. 
		Includes training, evaluation, model checkpoints, and prediction.

	./reading
		Reading materials related to the Deep Virtual Rapport Agent project.

	./smartbody_scripts
		Trial scripts used to test the communication with the Smartbody. 
