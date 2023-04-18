import cv2

# Load the video feed from the traffic camera
cap = cv2.VideoCapture('traffic_cam_feed.mp4')

# Define the object detection model
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # Apply the object detection model to the frame
    mask = object_detector.apply(frame)

    # Remove noise from the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the detected objects
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Traffic Management', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video feed and close all windows
cap.release()
cv2.destroyAllWindows()
