#!/bin/bash
#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Convert audio to FLAC format and resample to 16kHz, for Google ASR speech-to-text (STT)
#######################################################################################################################


input_dir=/home/ICT2000/jondras/dvra_datasets/mimicry/audio/audio_separated_48kHz
output_dir=/home/ICT2000/jondras/dvra_datasets/mimicry/audio/audio_separated_16kHz_flac

echo "Input path: ${input_dir}"
echo "Output path: ${output_dir}"

cnt=0
echo "Filepath, Filename, Time taken (seconds):"
for audiofile in ${input_dir}/*.wav
do
   cnt=$((cnt+1))
   idd=$(basename -- $audiofile)
   start_time=$(date +%s)

   sox -G "$audiofile" --channels 1 --rate 16000 ${output_dir}/${idd/.wav/.flac}
   
   end_time=$(($(date +%s)-start_time))
   echo $audiofile","$idd","$end_time
done

echo ""
echo "Converted ${cnt} audio files."
