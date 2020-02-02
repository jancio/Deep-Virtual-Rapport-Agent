#!/bin/bash
#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Downsample mono audio to 16 kHz for audio feature extraction and Google ASR (requires 16kHz audio sampling rate)
#######################################################################################################################


input_dir=/home/ICT2000/jondras/dvra_datasets/mimicry/audio/audio_separated_48kHz
output_dir=/home/ICT2000/jondras/dvra_datasets/mimicry/audio/audio_separated_16kHz

echo "Input path: ${input_dir}"
echo "Output path: ${output_dir}"

cnt=0
echo "Filepath, Filename, Time taken (seconds):"

for f in ${input_dir}/*.wav
do
   cnt=$((cnt+1))
   idd=$(basename -- $f)
   start_time=$(date +%s)

   sox $f -r 16000 ${output_dir}/${idd}
   
   end_time=$(($(date +%s)-start_time))
   echo $f","$idd","$end_time
done

echo ""
echo "Downsampled ${cnt} audio files."
