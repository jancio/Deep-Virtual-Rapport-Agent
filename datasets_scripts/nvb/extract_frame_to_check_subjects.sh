#!/bin/bash

#######################################################################################################################
# Project: Deep Virtual Rapport Agent (data preprocessing)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Extract 1 frame from each video to check whether all subjects differ
# (i.e., to manually verify that each video contains a different subject)
#######################################################################################################################


input_dir=~/dvra_datasets/nvb/original_data/videos
output_dir=~/dvra_datasets/nvb/original_data/sample_frames

mkdir ${output_dir}
cnt=0

for f in ${input_dir}/*.mp4
do
   cnt=$((cnt+1))
   echo "Processing "$f
   mplayer -vo jpeg:outdir=$output_dir/$cnt -frames $cnt $f
done

echo ""
echo "Extracted frames from ${cnt} videos."
