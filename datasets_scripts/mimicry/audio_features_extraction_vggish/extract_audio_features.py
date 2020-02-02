#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Extract audio features from the Mimicry dataset using VGGish
#     Rewritten based on vggish_inference_demo.py and kalins_extract_audio_features.py.
#     Finally, not used since for the real-time embedding into the Multisense system OpenSMILE might be better.
#######################################################################################################################


import time
import glob
import argparse
import numpy as np
from scipy.io import wavfile
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default='./vggish_model.ckpt')
parser.add_argument('--input_path', default='/home/ICT2000/jondras/dvra_datasets/mimicry/audio/audio_separated_16kHz')
parser.add_argument('--output_path', default='/home/ICT2000/jondras/dvra_datasets/mimicry/audio_features')
args = parser.parse_args()

start_time = time.time()
cnt = 0
for audio_filepath in sorted(glob.glob(f'{args.input_path}/*.wav')):

    audio_basename = audio_filepath.split('/')[-1].split('.')[0]
    print(f'Extracting features for {audio_basename}')

    wav_rate, wav_samples = wavfile.read(audio_filepath)
    print(f'\t{wav_rate} {len(wav_samples)}')
    if len(wav_samples) < wav_rate:
        wav_samples = np.pad(wav_samples, (0, wav_rate - len(wav_samples)), 'constant')

    samples = vggish_input.waveform_to_examples(wav_samples, wav_rate)

    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, args.model_file)
    
        samples_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        print(time.time() - start_time)
        [features] = sess.run([features_tensor], feed_dict={samples_tensor: samples})
        print(time.time() - start_time)

    print(f'\t{features.shape}')
    # print(f'\t{features}')
    np.save(f'{args.output_path}/{audio_basename}.npy', features)
    # break



# flags = tf.app.flags

# flags.DEFINE_string(
#         'wav_file', None,
#         'Path to a wav file. Should contain signed 16-bit PCM samples. '
#         'If none is provided, a synthetic sound is used.')

# flags.DEFINE_string(
#         'checkpoint', 'vggish_model.ckpt',
#         'Path to the VGGish checkpoint file.')

# flags.DEFINE_string(
#         'pca_params', 'vggish_pca_params.npz',
#         'Path to the VGGish PCA parameters file.')

# flags.DEFINE_string(
#         'tfrecord_file', None,
#         'Path to a TFRecord file where embeddings will be written.')

# FLAGS = flags.FLAGS
'''
    examples_batch = vggish_input.wavfile_to_examples(wav_file)
    print(examples_batch)

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                                                 feed_dict={features_tensor: examples_batch})
        print(embedding_batch)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        print(postprocessed_batch)

        # Write the postprocessed embeddings as a SequenceExample, in a similar
        # format as the features released in AudioSet. Each row of the batch of
        # embeddings corresponds to roughly a second of audio (96 10ms frames), and
        # the rows are written as a sequence of bytes-valued features, where each
        # feature value contains the 128 bytes of the whitened quantized embedding.
        seq_example = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(
                        feature_list={
                                vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                                        tf.train.FeatureList(
                                                feature=[
                                                        tf.train.Feature(
                                                                bytes_list=tf.train.BytesList(
                                                                        value=[embedding.tobytes()]))
                                                        for embedding in postprocessed_batch
                                                ]
                                        )
                        }
                )
        )
        print(seq_example)
        if writer:
            writer.write(seq_example.SerializeToString())

    if writer:
        writer.close()

# POST-PROCESSING?
# WHY PADDING?
'''