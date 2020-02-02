#By Leili Tavabi - Feb 2019

#from google.cloud import speech
#from google.cloud import speech_v1p1beta1 as speech
#from google.cloud.speech import enums
#from google.cloud.speech import types
from google.cloud import speech_v1p1beta1 as speech 
from google.cloud.speech_v1p1beta1 import enums
import pandas as pd 
import unicodedata
import argparse
import io, os 

#export GOOGLE_APPLICATION_CREDENTIALS="./service-account.json"

# [START speech_transcribe_async]
def transcribe_file(speech_file):
    """Transcribe the given audio file asynchronously."""
    client = speech.SpeechClient()
    print "client created"

    # [START speech_python_migration_async_request]
    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US')
        #enable_speaker_diarization=True, 
        #diarization_speaker_count=2)

    # [START speech_python_migration_async_response]
    operation = client.long_running_recognize(config, audio)
    # [END speech_python_migration_async_request]

    print('Waiting for operation to complete...')
    response = operation.result(timeout=90)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u'Transcript: {}'.format(result.alternatives[0].transcript))
        print('Confidence: {}'.format(result.alternatives[0].confidence))
    # [END speech_python_migration_async_response]
# [END speech_transcribe_async]


# [START speech_transcribe_async_gcs]
def transcribe_gcs(gcs_uri):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    audio = types.RecognitionAudio(uri=gcs_uri)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code='en-US',
        enable_speaker_diarization=True, 
        diarization_speaker_count=2)

    operation = client.long_running_recognize(config, audio)

    print('Waiting for operation to complete...')
    response = operation.result()

    '''
    result = response.results[-1]
    words_info = result.alternatives[0].words

    # Printing out the output:
    for word_info in words_info:
       print("word: '{}', speaker_tag: {}".format(word_info.word, word_info.speaker_tag))

    '''

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u'Transcript: {}'.format(result.alternatives[0].transcript))
        print('Confidence: {}'.format(result.alternatives[0].confidence))
# [END speech_transcribe_async_gcs]

def save_result_sentences(response, sent_path, file_name):

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

    print "Saving Dump to File... "
    file_path = os.path.join(dump_path, file_name + '_dump.txt')

    with open(file_path, 'w') as f: 
        print>> f, response
    

    

def transcribe_gcs_diarization(file_name):

    
    client = speech.SpeechClient()

    '''
    speech_file = 'resources/commercial_mono.wav'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    '''


    audio = speech.types.RecognitionAudio(uri=file_name)

    config = speech.types.RecognitionConfig(
        #encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code='en-GB',
        enable_speaker_diarization=True,
        diarization_speaker_count=2)


    operation = client.long_running_recognize(config, audio)

    print('Waiting for operation to complete...')
    #response = client.recognize(config, audio)
    response = operation.result()

    #result = response.results

    #words_info = result.alternatives[0].words

    #for word_info in words_info:
    #    print("word: '{}', speaker_tag: {}".format(word_info.word, word_info.speaker_tag))
    
    return response

def run_and_save_transcript(file_name, sentence_path, word_path, dump_path):

    #response = transcribe_gcs_diarization('gs://dcaps/WoZ_audio/'+file_name+'_AUDIO.flac')
    response = transcribe_gcs_diarization('gs://dcaps/'+file_name+'.flac')
    #response = transcribe_gcs_diarization('gs://dcaps/'+file_name+"_AUDIO.flac")
    
    save_result_sentences(response, sentence_path, file_name+"_G_ASR")
    save_result_words(response, word_path, file_name+"_G_ASR")
    save_result_dumps(response, dump_path, file_name+"_G_ASR")


if __name__ == '__main__':
    
    '''
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'file_name', help='File or GCS path for audio file to be recognized')
    
    args = parser.parse_args()
    '''
    
    sentence_path, word_path, dump_path = 'out/Janko/sentences', 'out/Janko/words', 'out/Janko/dumps'
    
    source_path = 'data/Janko'
    audio_files = os.listdir(source_path)
    audio_files.sort()

    for f in audio_files: 
        if f.endswith('.flac'): 
            #if f >= "365_AUDIO.flac":
            print 'Processing audio file number ', f, ' ...'
            #run_and_save_transcript(f[:f.index("_")], sentence_path, word_path, dump_path)
            run_and_save_transcript(f[:f.index('.')], sentence_path, word_path, dump_path)



    
    
