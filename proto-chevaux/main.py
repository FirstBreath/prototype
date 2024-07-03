import cv2
import numpy as np
import threading
import time
from object_detection import ObjectDetection

od = ObjectDetection()

cap = cv2.VideoCapture("cheval.mp4")

# Create a named window with the ability to resize
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

# Resize the window to specific dimensions
cv2.resizeWindow('Frame', 1600, 1200)  # Adjust the size as needed

# Global variables for threading
frame_resized = None
boxes = []
stop_event = threading.Event()  # Event to signal the thread to stop

def process_frame():
    global frame_resized, boxes
    while not stop_event.is_set():
        if frame_resized is not None:
            (class_ids, scores, boxes) = od.detect(frame_resized)

# Start the processing thread
processing_thread = threading.Thread(target=process_frame)
processing_thread.daemon = True
processing_thread.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            # If we reach the end of the video, reset to the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize the frame to be 2 times larger for display
        frame_resized = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))

        # Draw the boxes on the resized frame
        for box in boxes:
            (x, y, w, h) = box
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 10)

        # Display the resized frame with detection boxes
        cv2.imshow("Frame", frame_resized)

        key = cv2.waitKey(10)
        if key == 27:  # Escape key
            break
except Exception as e:
    print(f"Exception: {e}")
finally:
    # Signal the thread to stop
    stop_event.set()
    processing_thread.join()  # Wait for the thread to finish
    cap.release()
    cv2.destroyAllWindows()
