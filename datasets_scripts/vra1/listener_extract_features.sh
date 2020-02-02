#!/bin/bash

#######################################################################################################################
# Project: Deep Virtual Rapport Agent (data preprocessing)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Extract listener vision features from the vra1 dataset using OpenFace
#
# Note: the scripts aggregate_all_*.sh need to be run first.
#######################################################################################################################


input_dir=~/dvra_datasets/vra1/listener_videos
output_dir=~/dvra_datasets/vra1/listener_openface_features
output_log=~/dvra_datasets/vra1/listener_feature_extraction_time_log.txt

echo "Input path: ${input_dir}"
echo "Output path: ${output_dir}"
echo ""

openface_dir=~/OpenFace/build/bin
cd $openface_dir
cnt=0


for f in ${input_dir}/*.mp4
do

   cnt=$((cnt+1))
   idd=$(basename -- $f)

   start_time=$(date +%s)

   ./FeatureExtraction -f $f -out_dir ${output_dir}

   # mv ${output_path}/${idd}'.csv' ${output_path}/${idd}'_FACE.csv'
   # rm ${output_path}/${idd}'_of_details.txt'
   # rm ${output_path}/${idd}'.avi'

   echo "Filename, Time taken (seconds):"
   end_time=$(($(date +%s)-start_time))
   echo $idd","$end_time
   echo $idd","$end_time >> $output_log
   echo ""
done

echo ""
echo "Extracted features from ${cnt} videos."
