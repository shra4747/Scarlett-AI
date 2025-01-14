from pytube import YouTube
import cv2
import numpy as np


# Step 1: Download the live stream video
def download_video(youtube_url, output_path="stream.mp4"):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    stream.download(filename=output_path)
    return output_path


download_video("https://youtu.be/vfQy-WtjwEM")