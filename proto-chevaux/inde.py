import cv2

# Load the video
video_path = 'los_angeles.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize the tracker
tracker = cv2.TrackerMIL.create() # You can use other trackers like TrackerKCF_create, TrackerMIL_create, etc.

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Select the bounding box around the object you want to track
bbox = cv2.selectROI('Frame', frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Frame')

# Initialize the tracker with the first frame and bounding box
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    ret, bbox = tracker.update(frame)

    if ret:
        # Draw the bounding box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

    # Display the frame
    cv2.imshow('Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
