import cv2
import numpy as np
import threading
import time
from object_detection import ObjectDetection

od = ObjectDetection()
url = '../assets/cheval.mp4'

cap = cv2.VideoCapture(url)

# Create named windows with the ability to resize
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('Masked ROI', cv2.WINDOW_NORMAL)
cv2.namedWindow('Extracted Shape', cv2.WINDOW_NORMAL)

# Resize the window for the frame to specific dimensions
cv2.resizeWindow('Frame', 1600, 1200)  # Adjust the size as needed

backSub = cv2.createBackgroundSubtractorMOG2()

# Global variables for threading
frame_resized = None
boxes = []
stop_event = threading.Event()  # Event to signal the thread to stop

# Get the video's frames per second (FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(1000 / fps)  # Frame interval in milliseconds

def process_frame():
    global frame_resized, boxes
    while not stop_event.is_set():
        if frame_resized is not None:
            (class_ids, scores, boxes) = od.detect(frame_resized)
        time.sleep(1)  # Run detection once per second

# Start the processing thread
processing_thread = threading.Thread(target=process_frame)
processing_thread.daemon = True
processing_thread.start()

try:
    while True:
        start_time = time.time()  # Record the start time

        ret, frame = cap.read()
        if not ret:
            # If we reach the end of the video, reset to the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize the frame to be 2 times larger for display
        frame_resized = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))

        masked_roi_frame = None  # Initialize the masked ROI frame

        # Draw the boxes on the resized frame
        for box in boxes:
            (x, y, w, h) = box
            roi = frame_resized[y:y+h, x:x+w]
        
            # Apply background subtraction to the ROI
            fgMask = backSub.apply(roi)
            
            # Convert the ROI to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply the mask to the grayscale ROI
            masked_roi = cv2.bitwise_and(gray_roi, gray_roi, mask=fgMask)
            
            # Convert the masked grayscale ROI back to BGR for displaying in the same window
            masked_roi_bgr = cv2.cvtColor(masked_roi, cv2.COLOR_GRAY2BGR)
            
            masked_roi_frame = masked_roi_bgr  # Update the masked ROI frame

            # Draw the bounding box on the original frame
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # Display the resized frame with detection boxes
        cv2.imshow("Frame", frame_resized)

        # Display the masked ROI in the separate window
        if masked_roi_frame is not None:
            cv2.imshow("Masked ROI", masked_roi_frame)

            # Find contours in the masked ROI
            contours, _ = cv2.findContours(masked_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a blank canvas to draw contours
            extracted_shape = np.zeros_like(masked_roi_bgr)

            # Draw contours on the blank canvas
            cv2.drawContours(extracted_shape, contours, -1, (0, 255, 0), 1)

            # Display the extracted shape in the third window
            cv2.imshow("Extracted Shape", extracted_shape)

        key = cv2.waitKey(frame_interval)
        if key == 27:  # Escape key
            break

        # Calculate the time to sleep to maintain the original frame rate
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        sleep_time = max(1, frame_interval - int(elapsed_time))
        time.sleep(sleep_time / 1000.0)  # Sleep to maintain the frame rate
except Exception as e:
    print(f"Exception: {e}")
finally:
    # Signal the thread to stop
    stop_event.set()
    processing_thread.join()  # Wait for the thread to finish
    cap.release()
    cv2.destroyAllWindows()
