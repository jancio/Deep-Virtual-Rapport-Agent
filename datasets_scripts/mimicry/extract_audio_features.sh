#!/bin/bash
#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Extract audio/speech features from the Mimicry dataset using OpenSMILE.
#     Extracts 2 sets of features: emobase and extended MFCC.
#######################################################################################################################


input_dir=/home/ICT2000/jondras/dvra_datasets/mimicry/audio/audio_separated_16kHz
output_dir_emobase=/media/jondras/KALINS_BKP/Janko_package/dvra_datasets/mimicry/audio_features/opensmile_emobase
output_dir_mfcc=/media/jondras/KALINS_BKP/Janko_package/dvra_datasets/mimicry/audio_features/opensmile_mfcc


echo "Input path: ${input_dir}"
echo "Output path: ${output_dir}"

cd /home/ICT2000/jondras/opensmile-2.3.0/config
cnt=0

for f in ${input_dir}/*.wav
do
    cnt=$((cnt+1))
    idd=$(basename -- $f)
    start_time=$(date +%s)
   
    # Extract emobase LLD and delta LLD
    ~/opensmile-2.3.0/SMILExtract -configfile emobase_csv.conf -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 -inputfile $f -lldcsvoutput ${output_dir_emobase}/${idd:0:-4}.csv
    
    # Pitch, intensity, loudness, log energy, MFCC, and deltas and delta-deltas of all these
    ~/opensmile-2.3.0/SMILExtract -configfile MFCC12_0_D_A_extra.conf -appendcsv 0 -timestampcsv 1 -headercsv 1 -inputfile $f -csvoutput ${output_dir_mfcc}/${idd:0:-4}.csv

    echo "Filepath, Filename, Time taken (seconds):"
    end_time=$(($(date +%s)-start_time))
    echo $f","$idd","$end_time
    echo ""
done

echo ""
echo "Audio features extracted from ${cnt} audio files."
