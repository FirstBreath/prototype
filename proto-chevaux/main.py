import cv2
import numpy as np
from object_detection import ObjectDetection

od = ObjectDetection()

cap = cv2.VideoCapture("test.mp4")

# Create a named window with the ability to resize
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

# Resize the window to specific dimensions
cv2.resizeWindow('Frame', 800, 600)  # Adjust the size as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()