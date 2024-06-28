from __future__ import print_function
import cv2 as cv
import argparse
from pytube import YouTube

# Function to download YouTube video
def download_video(url, output_path='video.mp4'):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4').first()
    stream.download(filename=output_path)
    return output_path

# Argument parser setup
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

# Create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# URL of the video to process
url = 'https://www.youtube.com/watch?v=Qaod9E2P9Zk'

# Download the video
video_path = download_video(url)

# Open the downloaded video
capture = cv.VideoCapture(video_path)
if not capture.isOpened():
    print('Unable to open: ' + video_path)
    exit(0)

# Process the video frames
while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # Update the background model
    fgMask = backSub.apply(frame)

    # Display frame number on the current frame
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Show the current frame and the foreground mask
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    # Break the loop if 'q' or 'ESC' is pressed
    keyboard = cv.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break

# Release the video capture object and close all OpenCV windows
capture.release()
cv.destroyAllWindows()
