#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Run Google ASR speech-to-text service
#     Save all results for words and sentences, and also the raw dumps as JSON.
#######################################################################################################################


from google.cloud import storage
from google.cloud import speech_v1p1beta1 as speech 
import pandas as pd 
import unicodedata
import io, os
import json
from google.protobuf.json_format import MessageToJson

#export GOOGLE_APPLICATION_CREDENTIALS="./Rapport-2f4c2bc0eb53.json"


def save_result_sentences(response, sent_path, file_name):

    if not os.path.exists(sent_path):
        os.makedirs(sent_path)
    print "Saving Sentences to File... "
    results = response.results

    file_path = os.path.join(sent_path, file_name+'.csv')

    text_arr, confidence_arr, start_time_arr, end_time_arr = [], [], [], []


    for result in results:

        transcript_str = unicodedata.normalize('NFKD', result.alternatives[0].transcript).encode('ascii', 'ignore')
        #transcript = str(result.alternatives[0].transcript)
        transcript = transcript_str
        confidence = result.alternatives[0].confidence
        
        start_time_seconds = result.alternatives[0].words[0].start_time.seconds
        start_time_nano = result.alternatives[0].words[0].start_time.nanos
        start_time = start_time_seconds + start_time_nano * 0.000000001


        end_time_seconds = result.alternatives[0].words[-1].end_time.seconds
        end_time_nano = result.alternatives[0].words[-1].end_time.nanos
        end_time = end_time_seconds + end_time_nano * 0.000000001

        text_arr.append(transcript)
        confidence_arr.append(round(confidence,3))
        start_time_arr.append(start_time)
        end_time_arr.append(end_time)


    cols = ['Start_Time', 'End_Time', 'Text', 'Confidence',]
    d = {"Start_Time": start_time_arr, "End_Time": end_time_arr, "Text": text_arr, "Confidence": confidence_arr, }
    df = pd.DataFrame(d, columns =cols ) 
    
    df.to_csv(file_path, sep=',', header=True, columns=cols, index=False)
        

def save_result_words(response, word_path, file_name):

    if not os.path.exists(word_path):
        os.makedirs(word_path)
    print "Saving Words to File... "
    results = response.results

    file_path = os.path.join(word_path, file_name+'_words.csv')

    word_arr, start_time_arr, end_time_arr, speaker_arr = [], [], [], []


    for result in results:

        words = result.alternatives[0].words
        for w in words: 

            word = unicodedata.normalize('NFKD', w.word).encode('ascii', 'ignore')

            start_time_seconds = w.start_time.seconds
            start_time_nano = w.start_time.nanos
            start_time = start_time_seconds + start_time_nano * 0.000000001


            end_time_seconds = w.end_time.seconds
            end_time_nano = w.end_time.nanos
            end_time = end_time_seconds + end_time_nano * 0.000000001

            speaker_tag = w.speaker_tag


            word_arr.append(word)
            start_time_arr.append(start_time)
            end_time_arr.append(end_time)
            speaker_arr.append(speaker_tag)

        word_arr.append("")
        start_time_arr.append("")
        end_time_arr.append("")
        speaker_arr.append("")


    cols = ['Start_Time', 'End_Time', 'Word', "Speaker_Tag"]
    d = {"Start_Time": start_time_arr, "End_Time": end_time_arr, "Word": word_arr, "Speaker_Tag": speaker_arr }
    df = pd.DataFrame(d, columns =cols ) 
    
    df.to_csv(file_path, sep=',', header=True, columns=cols, index=False)


def save_result_dumps(response, dump_path, file_name):

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    print "Saving Dump to File... "
    file_path = os.path.join(dump_path, file_name+'_dump.json')

    with open(file_path, 'w') as f: 
        f.write(MessageToJson(response))
    

def transcribe_gcs_diarization(file_name):

    
    client = speech.SpeechClient()

    '''
    speech_file = 'resources/commercial_mono.wav'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    '''

    audio = speech.types.RecognitionAudio(uri=file_name)

    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code='en-GB',
        enable_speaker_diarization=True,
        diarization_speaker_count=2,
        # model='video', # only with language_code='en-US',
        # use_enhanced=True
    )

    operation = client.long_running_recognize(config, audio)

    print('\tWaiting for operation to complete...')
    #response = client.recognize(config, audio)
    response = operation.result()

    #result = response.results

    #words_info = result.alternatives[0].words

    #for word_info in words_info:
    #    print("word: '{}', speaker_tag: {}".format(word_info.word, word_info.speaker_tag))
    
    return response


def run_and_save_transcript(input_file_path, output_path_prefix):

    response = transcribe_gcs_diarization(input_file_path)
    
    file_name = input_file_path.split('/')[-1].split('.')[-2]
    save_result_sentences(response, output_path_prefix+"/sentences", file_name+"_G_ASR")
    save_result_words(response, output_path_prefix+"/words", file_name+"_G_ASR")
    save_result_dumps(response, output_path_prefix+"/dumps", file_name+"_G_ASR")


if __name__ == '__main__':
    
    output_path_prefix = 'mimicry/google_stt_outputs'

    # Get input paths
    storage_client = storage.Client()
    blobs = storage_client.list_blobs('rapport', prefix='mimicry/audio/audio_separated_16kHz_flac/')
    audio_files = sorted(['gs://rapport/'+blob.name for blob in blobs])

    for f in audio_files: 
        if f.endswith('.flac'):
            print 'Processing audio file ', f, ' ...'
            run_and_save_transcript(f, output_path_prefix)
