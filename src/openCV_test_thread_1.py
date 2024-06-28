import cv2
import threading
import numpy as np


# Function to detect circles
def detect_circles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    return cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                            param1=50, param2=30, minRadius=15, maxRadius=100)


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
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    threads = []
    for (x, y, r) in circles:
        t = threading.Thread(target=cv2.circle, args=(frame, (x, y), r, (0, 255, 0), 4))
        t2 = threading.Thread(target=cv2.rectangle, args=(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1))
        t2.start()
        t.start()
        threads.append(t)
        threads.append(t2)
    for thread in threads:
        thread.join()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
