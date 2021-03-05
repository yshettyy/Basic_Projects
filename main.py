import os
from multiprocessing import Process
from src.audio import audiorecord
from src.video import videorecord
import numpy as np
import pandas as pd
import subprocess
from scipy.io import wavfile
import skvideo.io


def filepath(audio,video):
    dir = "output"
    audiopath = dir + os.sep + audio
    videopath = dir + os.sep + video
    return audiopath, videopath, dir


def merge_files(audio, video, filename):
    """
    :param audio: file path to audio
    :param video: file path to video
    :param filename: merged file name
    :return: returns synchronized file of audio and video
    """
    audio, video, dir_ = filepath(audio, video)
    path = dir_+os.sep+filename+".mkv"
    cmd = 'ffmpeg -y -i {}  -r 30 -i {} -filter:a aresample=async=1 -c:a flac -c:v copy {}'.format(audio,video, path)
    subprocess.call(cmd, shell=True)  # merging of the files has done
    return print('merged file has stored at {}'.format(path))


def arrayex(merged_video):
    """
    :param merged_video: name of the merged video file
    :return: extracted ndarray of the audio and vdeo file
    """
    path_ = "output/"+merged_video
    video_data = skvideo.io.vread(path_)
    return video_data


if __name__ == '__main__':
    try:
        p1 = Process(target=videorecord)
        p2 = Process(target=audiorecord)
        p1.start()
        p2.start()
        p1.join(timeout=30) # timeout in 30 seconds. it is changeable.
        p2.join(timeout=30)
        merge_files(audio="audio.wav", video="video.avi",filename="final_output")
        if os.path.exists("output/final_output.mkv"):
            video_array = arrayex(merged_video="final_output.mkv")
            print("numpy array extracted.")
        print('#' * 80)
        print('press Ctrl+C to stop the recording')
        print('#' * 80)
    except KeyboardInterrupt:
        print('\nRecording finished')
