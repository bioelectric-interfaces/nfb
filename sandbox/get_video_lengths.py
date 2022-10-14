import cv2
import datetime
import os

# folder path
dir_path = r'C:\Users\christ\Desktop\tacs_videos'

# list file and directories
file_list = os.listdir(dir_path)
print(file_list)

# create video capture object
video_lengths = []
for f in file_list:
    data = cv2.VideoCapture(os.path.join(dir_path, f))

    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)

    # calculate duration of the video
    seconds = round(frames / fps)
    video_lengths.append(seconds)
print(video_lengths)
sum(video_lengths)
