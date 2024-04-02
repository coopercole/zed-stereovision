import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
import cv2.aruco as aruco
import fractions
import csv
import os
import datetime

def get_point_cloud_depth(point_cloud_value, x, y):
    if math.isfinite(point_cloud_value[2]):
        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                            point_cloud_value[1] * point_cloud_value[1] +
                            point_cloud_value[2] * point_cloud_value[2])
        
        # Print the point cloud value
        print(f"Distance to Target: {mm_to_feet_inches_fractions(float(distance))}, {round(distance, 2)} mm")
        return mm_to_feet_inches_fractions(float(distance)), round(distance, 2)
    else:
        print(f"The distance can not be computed at {{{x};{y}}}")
        return None

def get_aruco_marker_orientation(image, aruco_dict, parameters, cameraMatrix, distCoeffs):
    #  --------------- NOT USED ----------------
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the image to a numpy array
    gray_image = gray_image.astype(np.uint8)

    # Detect the aruco markers in the frame
    corners, ids, rejected = aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)

    # If any markers are detected
    if ids is not None:
        # Estimate the pose of the ArUco marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)

        # Get the rotation vector and translation vector of the first detected marker
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]

        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Get the orientation angles from the rotation matrix
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

        # Convert the angles from radians to degrees
        yaw_degrees = round(math.degrees(yaw), 2)
        pitch_degrees = round(math.degrees(pitch), 2)
        roll_degrees = round(math.degrees(roll), 2)

        # Print the orientation angles
        if(PRINT_DATA): print(f"Orientation angles of the ArUco marker: Yaw: {yaw_degrees}°, Pitch: {pitch_degrees}°, Roll: {roll_degrees}°")

        # Draw the orientation vectors on the image
        image = cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.1)

        # Add text labels for yaw, pitch, and roll
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        text_thickness = 2
        text_scale = 0.8
        x, y = 50, 50

        # Yaw
        yaw_text = f"Yaw: {yaw_degrees}"
        yaw_text_position = (x + 20, y + 20)
        cv2.putText(image, yaw_text, yaw_text_position, font, text_scale, text_color, text_thickness)

        # Pitch
        pitch_text = f"Pitch: {pitch_degrees}"
        pitch_text_position = (x + 20, y + 50)
        cv2.putText(image, pitch_text, pitch_text_position, font, text_scale, text_color, text_thickness)

        # Roll
        roll_text = f"Roll: {roll_degrees}"
        roll_text_position = (x + 20, y + 80)
        cv2.putText(image, roll_text, roll_text_position, font, text_scale, text_color, text_thickness)

        # Display the image with the orientation vectors
        cv2.imshow("Image with Orientation Vectors", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return yaw_degrees, pitch_degrees, roll_degrees

def get_aruco_target_yaw(image, aruco_dict, parameters, cameraMatrix, distCoeffs):
    # USES THE YAW ANGLE OF THE ARUCO MARKER TO DETERMINE THE YAW ANGLE OF THE TARGET,
    # THIS ISN'T GREAT, ARUCO MARKER DOESN"T SHOW SQUARE VERY WELL AND ORIENTATION CALCULATION ISN'T GREAT
    # ------------------ NOT USED ------------------ 

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the image to a numpy array
    gray_image = gray_image.astype(np.uint8)

    # Detect the aruco markers in the frame
    corners, ids, rejected = aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)

    # If any markers are detected
    if ids is not None:
        # Estimate the pose of the ArUco marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)

        # Get the rotation vector and translation vector of the first detected marker
        rvec = rvecs[0][0]
        # tvec = tvecs[0][0] ## not used

        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Get what I think is the yaw from the rotation matrix
        yaw = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        # Convert the angles from radians to degrees
        yaw_degrees = round(math.degrees(yaw), 2)

        # Print the orientation angles
        if(PRINT_DATA): print(f"Yaw angle of the ArUco marker: {yaw_degrees}°")

        return yaw_degrees

    return None

def get_camera_orientation(camera):


    # ------------------------------------------- THIS SHIT DOESN"T WORK -------------------------------------------



    sensors_data = sl.SensorsData()

    # orientation = zed_imu.get_pose().get_rotation_matrix()

    rotation = sensors_data.get_imu_data().get_pose().get_euler_angles()
   
    # Get the rotation matrix of the camera
    # rotation_matrix = np.array(camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.get_rotation_matrix())
    rotation_matrix = np.array(sensors_data.get_imu_data().get_pose().get_rotation_matrix())

    # Get the rotation matrix of the aruco tag
    aruco_rotation_matrix = np.array([[math.cos(math.radians(180)), 0, math.sin(math.radians(180))],
                                      [0, 1, 0],
                                      [-math.sin(math.radians(180)), 0, math.cos(math.radians(180))]])

    # Calculate the relative rotation matrix
    relative_rotation_matrix = np.matmul(rotation_matrix, aruco_rotation_matrix)

    # Convert the relative rotation matrix to euler angles
    relative_euler_angles = cv2.RQDecomp3x3(relative_rotation_matrix)[0]

    # Get the yaw, pitch, and roll angles
    yaw = relative_euler_angles[1]
    pitch = relative_euler_angles[0]
    roll = relative_euler_angles[2]

    # Convert the angles from radians to degrees
    yaw_degrees = math.degrees(yaw)
    pitch_degrees = math.degrees(pitch)
    roll_degrees = math.degrees(roll)

    # Print the orientation angles
    print(f"IMU Orientation relative to ArUco tag: Yaw: {yaw_degrees}°, Pitch: {pitch_degrees}°, Roll: {roll_degrees}°")


    print(f"IMU Orientation: {rotation}")


    # Display the IMU orientation quaternion
    # zed_imu_pose = sl.Transform()
    # ox = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[0], 3)
    # oy = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[1], 3)
    # oz = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[2], 3)
    # ow = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[3], 3)
    # print("IMU Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))




def find_aruco_marker(image, aruco_dict, parameters, cameraMatrix, distCoeffs):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # # Convert the image to a numpy array
    # gray = gray.astype(np.uint8)

    # Detect the aruco markers in the frame
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # If any markers are detected
    if ids is not None:
        # Get the center coordinates of the first detected marker
        marker_center = np.mean(corners[0][0], axis=0)
        x = int(marker_center[0])
        y = int(marker_center[1])
        if PRINT_DATA: print(f"Center of the aruco marker: {{{x};{y}}}")
        # Draw a rectangle around the detected marker
        cv2.rectangle(gray, (x-10, y-10), (x+10, y+10), (0, 255, 0), 2)
       
        # Display the image with the detected marker
        if SHOW_IMAGE:
            cv2.imshow("Image", gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return x, y
    else:
        print("No ArUco marker detected.")
        return None