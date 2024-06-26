import cv2
import numpy as np

# Function to detect circles
def detect_circles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=15, maxRadius=100)
    return circles

# Open a video capture object

# url = 'rtsp://delenclosnathan.fr:8554/test'

cap = cv2.VideoCapture(0)  # 0 to use the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Detect circles in the frame
    circles = detect_circles(frame)

    # Draw the detected circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
