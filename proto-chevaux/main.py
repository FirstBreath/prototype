import cv2
import numpy as np
import threading
import time
from object_detection import ObjectDetection

od = ObjectDetection()

cap = cv2.VideoCapture("test1.mp4")

# Create a named window with the ability to resize
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

# Resize the window to specific dimensions
cv2.resizeWindow('Frame', 1600, 1200)  # Adjust the size as needed

# Global variables for threading
frame_resized = None
boxes = []

def process_frame():
    global frame_resized, boxes
    while True:
        if frame_resized is not None:
            (class_ids, scores, boxes) = od.detect(frame_resized)

# Start the processing thread
processing_thread = threading.Thread(target=process_frame)
processing_thread.daemon = True
processing_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to be 2 times larger for display
    frame_resized = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))

    # Draw the boxes on the resized frame
    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 5)

    # Display the resized frame with detection boxes
    cv2.imshow("Frame", frame_resized)

    key = cv2.waitKey(30)
    if key == 27:  # Escape key
        break

cap.release()
cv2.destroyAllWindows()
