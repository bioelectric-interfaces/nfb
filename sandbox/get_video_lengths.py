import cv2
import os
import random

def get_video_lengths(dir_path, file_list):
    video_lengths = {}
    for f in file_list:
        data_path = os.path.join(dir_path, f)
        data = cv2.VideoCapture(data_path)

        # count the number of frames
        frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = data.get(cv2.CAP_PROP_FPS)

        # calculate duration of the video
        seconds = round(frames / fps)
        video_lengths[data_path] = (seconds)
    return video_lengths

    # get a random allocation of lengths that are close to the desired duration

if __name__ == '__main__':# desired duration
    tacs_duration = 15*60

    # folder path
    dir_path = r'C:\Users\christ\Desktop\tacs_videos'
    # list file and directories
    file_list = os.listdir(dir_path)
    print(file_list)
    video_lengths = get_video_lengths(dir_path, file_list)


    print(video_lengths)

    items = list(video_lengths.items())  # List of tuples of (key,values)
    random.shuffle(items)
    for key, value in items:
        print(key, ":", value)
    sum(video_lengths)
