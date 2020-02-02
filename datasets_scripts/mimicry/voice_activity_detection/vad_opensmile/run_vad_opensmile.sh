#!/bin/bash
#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Run OpenSMILE Voice Activity Detection (VAD) on single-channel audio from Mimicry dataset
#######################################################################################################################


input_dir=/home/ICT2000/jondras/dvra_datasets/mimicry/audio/audio_separated_16kHz
output_dir=/home/ICT2000/jondras/dvra_datasets/mimicry/voice_activity_detection/vad_opensmile

echo "Input path: ${input_dir}"
echo "Output path: ${output_dir}"

cd /home/ICT2000/jondras/opensmile-2.3.0/scripts/vad
cnt=0

for f in ${input_dir}/*.wav
do
   cnt=$((cnt+1))
   idd=$(basename -- $f)
   start_time=$(date +%s)
   
         
   ~/opensmile-2.3.0/SMILExtract -C vad_opensource.conf -I $f -O ${output_dir}/vad_os_${idd:0:-4}.csv

   echo "Filepath, Filename, Time taken (seconds):"
   end_time=$(($(date +%s)-start_time))
   echo $f","$idd","$end_time
   echo ""
done

echo ""
echo "VAD run on ${cnt} audio files."
