#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Given original video file from Mimicry database and annotated vision features (csv), generate a new video file with 
# color annotations of head gestures (nod, shake, tilt), smile, gaze away, voice activity, and turn-taking.
#######################################################################################################################

import glob
import os
import pandas as pd
import numpy as np
import cv2


input_videos_dir = '/home/ICT2000/jondras/dvra_datasets/mimicry/original_data'
input_annotations_dir = '/home/ICT2000/jondras/dvra_datasets/mimicry/vision_features/annotated_features'
output_videos_dir = '/home/ICT2000/jondras/dvra_datasets/mimicry/generated_videos/fully_annotated_videos'

if not os.path.exists(output_videos_dir):
    os.makedirs(output_videos_dir)

# Iterate over annotated csv feature files
for input_annotations_file in sorted(glob.glob(f'{input_annotations_dir}/*.csv')):

    video_name = input_annotations_file.split('/')[-1][:-4]
    video_name_split = video_name.split('_')
    sessid = video_name_split[1]
    pid = video_name_split[2]
    print(video_name)

    # Continue only for some session ids
#     if int(sessid) < 52:
#         continue

    input_video_file = glob.glob(f'{input_videos_dir}/sessid_{sessid}/Sessions/{int(sessid)}/*_{pid}_FaceFar2_*.avi')[0]
    output_video_file = f'{output_videos_dir}/{video_name}.avi'

    df = pd.read_csv(input_annotations_file)
    cap = cv2.VideoCapture(input_video_file)

    # From mimicry db paper
    #     Frame rate: 58.047736
    #     Resolution: 1024x1024
    #     Codec: H264 - MPEG-4 AVC (part 10) (H264)
    
    # Set manually
    # input_video_frame_rate = 58.047736
    # dt = 1 / input_video_frame_rate
    # input_video_shape = (1024, 1024)

    # Set automatically
    # print(int(cap.get(cv2.CAP_PROP_FRAME_RATE)))
    # print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    # print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # print(cap.get(cv2.CAP_PROP_FOURCC))
    input_video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    dt = 1 / input_video_frame_rate
    input_video_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f'\tFPS: {input_video_frame_rate:.6f}\t\tFrame size: {input_video_shape}\n')
    print('\t Timestamp || Nod | Shake | Tilt || Smile | Gaze away | Voice active | Take turn ')
    print('\t-----------||-----|-------|------||-------|-----------|--------------|-----------')

    # MJPG / XVID / DIVX
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video_writer = cv2.VideoWriter(output_video_file, fourcc, input_video_frame_rate, 
                                       input_video_shape, isColor=True)

    # Drawing settings for 2 boxes (first one: head gesture annotations; second one: other annotations) 
    # For color combinations see https://en.wikipedia.org/wiki/Web_colors
    text_x_start = int(input_video_shape[0] / 70.)
    text_y_spacing = int(input_video_shape[1] / 25.)
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = 1
    text_thickness = 3
    text_line_type = cv2.LINE_AA
    box_1_coords = np.array([
        [0.5*text_x_start, 3.3*text_y_spacing], 
        [16*text_x_start,  3.3*text_y_spacing], 
        [16*text_x_start,  0.1*text_y_spacing], 
        [0.5*text_x_start, 0.1*text_y_spacing]
    ], dtype=int)
    box_2_coords = np.array([
        [0.5*text_x_start, 7.8*text_y_spacing], 
        [16*text_x_start,  7.8*text_y_spacing], 
        [16*text_x_start,  3.6*text_y_spacing], 
        [0.5*text_x_start, 3.6*text_y_spacing]
    ], dtype=int)

    timestamp = 0.
    take_turn_start_time = -np.inf
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            
            # Take annotation for the current timestamp
            future_rows = df[df[' timestamp'] >= timestamp]
            if len(future_rows) > 0:
                current_row = future_rows.iloc[0]
                is_nod = int(current_row['nod'])
                is_shake = int(current_row['shake'])
                is_tilt = int(current_row['tilt'])
                is_smile = int(current_row['smile'])
                is_gaze_away = int(current_row['gaze_away'])
                is_voice_active = int(current_row['voice_active'])
                is_take_turn = int(current_row['take_turn'])
                # For testing
                # is_nod = 1
                # is_shake = 1
                # is_tilt = 1
                # is_smile = 1
                # is_gaze_away = 1
                # is_voice_active = 1
                # is_take_turn = 1
                
                # Annotate frame
                cv2.fillConvexPoly(frame, box_1_coords, (200, 200, 200))
                cv2.rectangle(frame, tuple(box_1_coords[0]), tuple(box_1_coords[2]), (25, 25, 25), thickness=3)
                if is_nod:
                    cv2.putText(frame, 'NOD',   (text_x_start, 1*text_y_spacing), text_font, text_size, (255, 0, 0), text_thickness, text_line_type)
                if is_shake:
                    cv2.putText(frame, 'SHAKE', (text_x_start, 2*text_y_spacing), text_font, text_size, (0, 255, 0), text_thickness, text_line_type)
                if is_tilt:
                    cv2.putText(frame, 'TILT',  (text_x_start, 3*text_y_spacing), text_font, text_size, (0, 0, 255), text_thickness, text_line_type)

                cv2.fillConvexPoly(frame, box_2_coords, (200, 200, 200))
                cv2.rectangle(frame, tuple(box_2_coords[0]), tuple(box_2_coords[2]), (25, 25, 25), thickness=3)
                if is_smile:
                    cv2.putText(frame, 'SMILE',        (text_x_start, int(4.4*text_y_spacing)), text_font, text_size, (32,  165, 218), text_thickness, text_line_type)
                if is_gaze_away:
                    cv2.putText(frame, 'GAZE AWAY',    (text_x_start, int(5.4*text_y_spacing)), text_font, text_size, (30,  105, 210), text_thickness, text_line_type)
                if is_voice_active:
                    cv2.putText(frame, 'VOICE ACTIVE', (text_x_start, int(6.4*text_y_spacing)), text_font, text_size, (19,  69,  139), text_thickness, text_line_type)
                # Exponential decay in color intensity for turn-taking annotations (decay constant is 0.25 sec)
                if is_take_turn:
                    take_turn_start_time = timestamp
                take_turn_intensity = np.exp(-(timestamp - take_turn_start_time) / 0.25)
                take_turn_color = tuple(int(200 - take_turn_intensity * (200 - c)) for c in (42,  42,  165))
                cv2.putText(frame, 'TAKE TURN',    (text_x_start, int(7.4*text_y_spacing)), text_font, text_size, take_turn_color, text_thickness, text_line_type)

                print(f'\t{timestamp:011.6f}    {is_nod}      {is_shake}      {is_tilt}        {is_smile}         {is_gaze_away}            {is_voice_active}             {is_take_turn}', end='\r')
            else:
                print(f'\t{timestamp:011.6f}   no more annotations found ...................')
            out_video_writer.write(frame)

            timestamp += dt
            # if timestamp > 10:
            #     break
        else:
            break

    # Release everything when finished
    cap.release()
    out_video_writer.release()

    print(f'\t{timestamp:011.6f}    {is_nod}      {is_shake}      {is_tilt}        {is_smile}         {is_gaze_away}            {is_voice_active}             {is_take_turn}\n')
    # break
    