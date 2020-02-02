#!/bin/bash

#######################################################################################################################
# Project: Deep Virtual Rapport Agent (data preprocessing)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Aggregate all annotations of listener nods into one folder
#######################################################################################################################


output_dir=~/dvra_datasets/vra1/listener_head_gesture_annotations/

cnt=0
for f in ./*transcriptions/*/*L.nod.eaf
do
	cnt=$((cnt+1))
	echo "${f}"

	cp $f $output_dir$(basename -- $f)
done

echo ""
echo "Copied ${cnt} annotations of listener nods."
