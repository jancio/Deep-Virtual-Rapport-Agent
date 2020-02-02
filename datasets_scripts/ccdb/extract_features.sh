#!/bin/bash

#######################################################################################################################
# Project: Deep Virtual Rapport Agent (data preprocessing)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Extract vision features from the ccdb dataset using OpenFace
#######################################################################################################################


input_dir=~/dvra_datasets/ccdb/original_data
output_dir=~/dvra_datasets/ccdb/openface_features
output_log=~/dvra_datasets/ccdb/feature_extraction_time_log.txt

rm ${output_log}

echo "Input path: ${input_dir}"
echo "Output path: ${output_dir}"
echo "Output log: ${output_log}"
echo ""
echo "filepath,filename,time" >> $output_log

openface_dir=~/OpenFace/build/bin
cd $openface_dir
cnt=0


# Extract features from the basic set of sessions
for f in ${input_dir}/*/*/*.mov
do

   cnt=$((cnt+1))
   idd=$(basename -- $f)

   start_time=$(date +%s)

   #./FeatureExtraction -f $f -out_dir ${output_dir} #-simscale 1.0 -simsize 224
   ./FeatureExtraction -f $f -out_dir ${output_dir} -2Dfp -3Dfp -pdmparams -pose -aus -gaze

   # mv ${output_path}/${idd}'.csv' ${output_path}/${idd}'_FACE.csv'
   # rm ${output_path}/${idd}'_of_details.txt'
   # rm ${output_path}/${idd}'.avi'

   echo "Filepath, Filename, Time taken (seconds):"
   end_time=$(($(date +%s)-start_time))
   echo $f","$idd","$end_time
   echo $f","$idd","$end_time >> $output_log
   echo ""
done

echo ""
echo "[basic set] Extracted features from ${cnt} videos."
echo ""


# Extract features from the extended set of sessions
for f in ${input_dir}/*/*.mp4
do

   cnt=$((cnt+1))
   idd=$(basename -- $f)

   start_time=$(date +%s)

   #./FeatureExtraction -f $f -out_dir ${output_dir} #-simscale 1.0 -simsize 224
   ./FeatureExtraction -f $f -out_dir ${output_dir} -2Dfp -3Dfp -pdmparams -pose -aus -gaze

   # mv ${output_path}/${idd}'.csv' ${output_path}/${idd}'_FACE.csv'
   # rm ${output_path}/${idd}'_of_details.txt'
   # rm ${output_path}/${idd}'.avi'

   echo "Filepath, Filename, Time taken (seconds):"
   end_time=$(($(date +%s)-start_time))
   echo $f","$idd","$end_time
   echo $f","$idd","$end_time >> $output_log
   echo ""
done

echo ""
echo "[extended set] Extracted features from ${cnt} videos."
