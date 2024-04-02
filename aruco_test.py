import cv2
import numpy as np
import cv2.aruco as aruco


# Open the webcam
cap = cv2.VideoCapture(0)

# Set the camera parameters
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create the aruco dictionary
# aruco_dict = aruco.Dictionary.create(aruco.DICT_6X6_250)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Create the aruco parameters
parameters = aruco.DetectorParameters()

# dummy camera matrix and distortion coefficients
cameraMatrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]])
distCoeffs = np.array([0, 0, 0, 0, 0])

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detect the aruco markers in the frame
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # If any markers are detected
    if ids is not None:
        # Draw the detected markers on the frame
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Get the depth measurement for the first detected marker
        # Replace 'marker_id' with the ID of the aruco tag you want to measure
        marker_id = 0
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)
        if marker_id in ids:
            index = np.where(ids == marker_id)
            rvec_marker = rvec[index[0][0]]
            tvec_marker = tvec[index[0][0]]
            depth = tvec_marker[0][2]

            # Print the depth measurement
            print("Depth:", depth)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()