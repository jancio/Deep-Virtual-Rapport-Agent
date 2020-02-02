#######################################################################################################################
# Project: Deep Virtual Rapport Agent (head gesture detector)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Given an original video file from a dataset and vision features (csv) annotated with head gestures using the developed 
# Head Gesture Detector, generate a new video file with color annotations of head gestures (nod, shake, tilt).
#######################################################################################################################


import glob
import pandas as pd
import numpy as np
import cv2


input_videos_dir = '/home/ICT2000/jondras/dvra_datasets/mimicry/original_data'
input_annotations_dir = '/home/ICT2000/jondras/dvra_datasets/mimicry/vision_features/annotated_features'
output_videos_dir = '/home/ICT2000/jondras/dvra_datasets/mimicry/generated_videos/hgd_annotated_videos'

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
    print('\t Timestamp || Nod | Shake | Tilt')
    print('\t-----------||-----|-------|-----')

    # MJPG / XVID / DIVX
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video_writer = cv2.VideoWriter(output_video_file, fourcc, input_video_frame_rate, 
                                       input_video_shape, isColor=True)

    # Drawing settings (x,y coordinates of bottom-left corner where text starts)
    text_x_start = int(input_video_shape[0] / 89.)
    text_y_spacing = int(input_video_shape[1] / 15.)
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = 2
    text_thickness = 5
    text_line_type = cv2.LINE_AA
    box_coords = np.array([
        [0.6*text_x_start, 3.2*text_y_spacing], 
        [25*text_x_start,  3.2*text_y_spacing], 
        [25*text_x_start,  0.1*text_y_spacing], 
        [0.6*text_x_start, 0.1*text_y_spacing]
    ], dtype=int)

    timestamp = 0.
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            
            # Take annotation for the current timestamp
            future_rows = df[df[' timestamp'] >= timestamp]
            if len(future_rows) > 0:
                current_row = future_rows.iloc[0]
                is_nod = (int(current_row['nod']) == 1)
                is_shake = (int(current_row['shake']) == 1)
                is_tilt = (int(current_row['tilt']) == 1)
                # For testing
                # is_nod = 1
                # is_shake = 1
                # is_tilt = 1
                
                # Annotate frame
                cv2.fillConvexPoly(frame, box_coords, (200, 200, 200))
                cv2.rectangle(frame, tuple(box_coords[0]), tuple(box_coords[2]), (25, 25, 25), thickness=3)
                if is_nod:
                    cv2.putText(frame, 'NOD',   (text_x_start, 1*text_y_spacing), text_font, text_size, (255, 0, 0), text_thickness, text_line_type)
                if is_shake:
                    cv2.putText(frame, 'SHAKE', (text_x_start, 2*text_y_spacing), text_font, text_size, (0, 255, 0), text_thickness, text_line_type)
                if is_tilt:
                    cv2.putText(frame, 'TILT',  (text_x_start, 3*text_y_spacing), text_font, text_size, (0, 0, 255), text_thickness, text_line_type)

                print(f'\t{timestamp:011.6f}    {is_nod:1}      {is_shake:1}      {is_tilt:1}', end='\r')
            else:
                print(f'\t{timestamp:011.6f}    no more annotations found ...................')
            out_video_writer.write(frame)

            timestamp += dt
            # if timestamp > 7:
            #     break
        else:
            break

    # Release everything when finished
    cap.release()
    out_video_writer.release()
    
    print(f'\t{timestamp:011.6f}    {is_nod:1}      {is_shake:1}      {is_tilt:1}\n')
