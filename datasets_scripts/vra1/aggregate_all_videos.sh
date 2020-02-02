#!/bin/bash

#######################################################################################################################
# Project: Deep Virtual Rapport Agent (data preprocessing)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Aggregate all listener videos into one folder
#######################################################################################################################


output_dir=~/dvra_datasets/vra1/listener_videos/

cnt=0
for f in ./*listener/*/*
do
	cnt=$((cnt+1))
	echo "${f}"

	cp $f $output_dir$(basename -- $f)
done

echo ""
echo "Copied ${cnt} videos of listener."
