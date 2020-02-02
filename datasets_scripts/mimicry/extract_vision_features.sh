#!/bin/bash
#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Extract vision features from the Mimicry dataset using OpenFace
# (optionally, also generate tracked videos by OpenFace)
#######################################################################################################################


# Session id to start with (when running the extraction in parallel)
start_sessid=0

input_dir=/home/ICT2000/jondras/dvra_datasets/mimicry/original_data
output_dir=/home/ICT2000/jondras/dvra_datasets/mimicry/vision_features/original_openface_features
# output_dir=/home/ICT2000/jondras/dvra_datasets/mimicry/generated_videos/openface_tracked_videos
output_log=/home/ICT2000/jondras/dvra_datasets/mimicry/vision_feature_extraction_time_log_${start_sessid}.txt

# rm ${output_log}

echo "Input path: ${input_dir}"
echo "Output path: ${output_dir}"
echo "Output log: ${output_log}"
echo ""
echo "filepath,sessid,filename,time" >> $output_log

openface_dir=~/OpenFace/build/bin
cd $openface_dir
cnt=0

for f in ${input_dir}/sessid*/Sessions/*/*FaceFar2*.avi
do
   # echo $f
   sessid=(${f:34:2})
   echo $sessid

   if [ $sessid -ge $start_sessid ]
   then
      cnt=$((cnt+1))
      idd=$(basename -- $f)

      start_time=$(date +%s)

      ./FeatureExtraction -f $f -out_dir ${output_dir} -2Dfp -3Dfp -pdmparams -pose -aus -gaze 
      # To get tracked videos
      # ./FeatureExtraction -f $f -out_dir ${output_dir} -tracked

      echo "Filepath, SESSID, Filename, Time taken (seconds):"
      end_time=$(($(date +%s)-start_time))
      echo $f","$sessid","$idd","$end_time
      echo $f","$sessid","$idd","$end_time >> $output_log
      echo ""
   fi

done

echo ""
echo "Extracted features from ${cnt} videos."
