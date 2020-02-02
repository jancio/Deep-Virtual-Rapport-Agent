import os

import tensorflow

import numpy
import argparse

from scipy.io import wavfile

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default='./preprocess/disclosure/dcaps/speech/AUDIOSET/model/model.ckpt')
parser.add_argument('--input_path', default='./preprocess/disclosure/dcaps/data/segment')
parser.add_argument('--output_path', default='./data/disclosure/dcaps/speech/AUDIOSET')
args = parser.parse_args()

folders = os.listdir(args.input_path)
for folder in folders:
  sub_folders = os.listdir(os.path.join(args.input_path, folder))
  
  for sub_folder in sub_folders:
    print('Extracting features for ', sub_folder)

    wav_name = os.path.join(args.input_path, folder, sub_folder, sub_folder + '.wav')

    wav_rate, wav_samples = wavfile.read(wav_name)
    if len(wav_samples)<wav_rate:
      wav_samples = numpy.pad(wav_samples, (0, wav_rate - len(wav_samples)), 'constant')

    samples = vggish_input.waveform_to_examples(wav_samples, wav_rate)

    with tensorflow.Graph().as_default(), tensorflow.Session() as session:
      vggish_slim.define_vggish_slim(training=False)
      vggish_slim.load_vggish_slim_checkpoint(session, args.model_file)
    
      samples_tensor = session.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
      features_tensor = session.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

      [features] = session.run([features_tensor], feed_dict={samples_tensor: samples})

    numpy.save(os.path.join(args.output_path, sub_folder + '.npy'), features)