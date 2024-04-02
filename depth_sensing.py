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

MILLIMETERS_TO_INCHES = 0.0393701
FRACTIONAL_PRECISION = 16
ZED2_BASELINE = 120  # mm
MARKER_SIZE = 0.3  # meters
PRINT_DATA = False
WRITE_TO_CSV = True
SHOW_IMAGE = True
PROCESS_IMAGE = True
SET_MIN_DIST = False
GET_YAW = True


def getMousePos(image):
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['x'] = x
            param['y'] = y
            param['event'] = event

    param = {'x': -1, 'y': -1, 'event': -1}
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", onMouse, param)
    cv2.imshow("Image", image)

    while param['event'] != cv2.EVENT_LBUTTONDOWN:
        cv2.waitKey(10)
    
    cv2.destroyWindow("Image")

    return param['x'], param['y']

def mm_to_feet_inches_fractions(mm):
    # Convert mm to inches
    inches = mm * MILLIMETERS_TO_INCHES
    # Calculate feet and remaining inches
    feet = int(inches // 12)
    remaining_inches = inches % 12

    # Convert remaining inches to fractions
    remaining_distance = remaining_inches - int(remaining_inches)

    numerator = round((remaining_distance * FRACTIONAL_PRECISION))
    denominator = FRACTIONAL_PRECISION
    fraction = fractions.Fraction(numerator, denominator) 
    
    # Format the result
    result = f"{feet}' {int(remaining_inches)}\" {fraction}"
    return result

def get_camera_matrix_and_distortion(camera):
    # Get the camera matrix and distortion coefficients from the ZED camera
    camera_info = camera.get_camera_information()
    # Get the camera matrix and distortion coefficients from the ZED camera
    cam_fx = camera_info.camera_configuration.calibration_parameters.left_cam.fx
    cam_fy = camera_info.camera_configuration.calibration_parameters.left_cam.fy
    cam_cx = camera_info.camera_configuration.calibration_parameters.left_cam.cx
    cam_cy = camera_info.camera_configuration.calibration_parameters.left_cam.cy
    cameraMatrix = np.array([[cam_fx, 0, cam_cx],
                            [0, cam_fy, cam_cy],
                            [0, 0, 1]])
    #  Get the distortion coefficients
    distCoeffs = camera_info.camera_configuration.calibration_parameters.left_cam.disto
    # # Convert to numpy array, come back to check this array, it apprears to cut off the last digit, expected values are [0.01, 0.1, 0.1, 0.1, 0.1]ish
    # distCoeffs = np.array(camera_info.camera_configuration.calibration_parameters.left_cam.disto) 

    # print the camera matrix and distortion coefficients
    if(PRINT_DATA):
        print(f"Camera Matrix: {cameraMatrix}")
        print(f"Distortion Coefficients: {distCoeffs}")

    return cameraMatrix, distCoeffs

def find_aruco_marker(image, aruco_dict, parameters, cameraMatrix, distCoeffs):

    if PROCESS_IMAGE:
        ## OUTSIDE HD2K 51.5 ft
        # y_start = 200
        # y_end = 600
        # x_start = 200
        # x_end = 800
        
        ## BEDROOM HD2K 10 ft
        y_start = 400
        y_end = 800
        x_start = 800
        x_end = 1400

        ## BEDROOM HD2K 10 ft wider view
        y_start = 400
        y_end = 800
        x_start = 400
        x_end = 1800

        ## ----- IMAGE PROCESSING BEGIN ----- ##
        # Crop the image and convert it to grayscale
        gray = cv2.cvtColor(image[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
        # Apply bilateral filter to the gray image
        bilateral_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        # if SHOW_IMAGE:
        #     cv2.imshow("Cropped, Gray and Filtered Image", bilateral_filtered)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        
        # # Apply thresholding to segment the black and white aruco marker, this is already done in detectMarkers
        # _, thresholded = cv2.threshold(bilateral_filtered, 100, 255, cv2.THRESH_BINARY)
        # if SHOW_IMAGE:
        #     cv2.imshow("Thresholded Image", thresholded)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # # Apply morphological operations to remove noise
        # kernel = np.ones((5, 5), np.uint8)
        # opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        # if SHOW_IMAGE:
        #     cv2.imshow("Morphological Operations", closing)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # Rename for clarity
        # processed_image = closing
        processed_image = bilateral_filtered
        ## IMAGE PROCESSING END

    else:
        # Convert the image to grayscale
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    # Detect the aruco markers in the cropped image, this also applies adaptive thresholding
    corners, ids, rejected = aruco.detectMarkers(processed_image, aruco_dict, parameters=parameters)
    
    # If any markers are detected
    if ids is not None:
        # Get the center coordinates of the first detected marker
        marker_center = np.mean(corners[0][0], axis=0)
        x = int(marker_center[0]) + x_start # add on the x_start to get the coordinates in the uncropped image
        y = int(marker_center[1]) + y_start # add on the y_start to get the coordinates in the uncropped image

        if GET_YAW:
            # Estimate the pose of the ArUco marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cameraMatrix, distCoeffs)

            # Get the rotation vector and translation vector of the first detected marker
            rvec = rvecs[0][0]
            tvec = tvecs[0][0] ## not used

            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Get the yaw from the rotation matrix
            yaw = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
            # Convert the angles from radians to degrees
            yaw_degrees = round(math.degrees(yaw), 2)
            print(f"Yaw angle of the ArUco marker: {yaw_degrees}°")

            marker_axis = cv2.drawFrameAxes(processed_image, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
            cv2.imshow("Processsed Image with Yaw", marker_axis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        if PRINT_DATA: print(f"Center of the aruco marker: {{{x};{y}}}")
        
        # Draw a rectangle around the detected marker in the image
        cv2.rectangle(image, (x-10, y-10), (x+10, y+10), (0, 255, 0), 2)
       
        # Display the image with the detected marker
        cv2.imshow("Raw Image with Detected Center", image[y-200:y+200, x-200:x+200])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # return the coordinates of the center of the aruco marker for depth sensing, and yaw angle
        return x, y
    else:
        print("---------NO ARUCO MARKER DETECTED---------")
        return None

def get_z_depth(depth, x, y):
    # Get the depth value at the detected point
    err, z = depth.get_value(x, y)

    z_inches = mm_to_feet_inches_fractions(z)
    print(f"Z Depth to Target: {z_inches}, {round(z, 2)} mm")

    return z

def get_x_distance(point_cloud_value):
    # THIS IS FLAWED, IT MEASURES THE DISTNACE IN THE FRAME, SO YAW AFFECTS IT

    # get the lateral distance to the target using the point cloud value of the center of the target
    x_distance = point_cloud_value[0] + ZED2_BASELINE/2 if point_cloud_value[0] <= 0 else point_cloud_value[0] - ZED2_BASELINE/2
    

    lateral_distance_inches = mm_to_feet_inches_fractions(x_distance)
    # print(f"X Distance to Target: {lateral_distance_inches}, {round(lateral_distance, 2)} mm")

    return round(x_distance, 2)

def get_camera_yaw(x, z):
    # Get the yaw angle of the camera using a triangle formed by the x and z coordinates
    yaw = math.atan2(-x, z)
    yaw_degrees = np.round(math.degrees(yaw),2)
    print(f"Yaw of the Camera: {yaw_degrees}°")
    return yaw_degrees

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

def write_to_csv(data):
    # Write the data to a CSV file
    if WRITE_TO_CSV:
        if not os.path.isfile('output.csv'):
            with open('output.csv', mode='a') as output_file:
                output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                output_writer.writerow(['Z [IMP]', 'Z [mm]', 'X [IMP]', 'X [mm]', 'Yaw [deg]'])
        with open('output.csv', mode='a') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow(data)

def main():

    # Define the camera parameters # NOT SURE IF THIS IS NEEDED
    # Set the camera parameters
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create the aruco dictionary, this is for the specfic aruco marker we are using
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
    # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    # Create the aruco parameters
    parameters = aruco.DetectorParameters()

    # Create a Camera object
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init_params.camera_resolution = sl.RESOLUTION.HD2K # Use HD2K video mode (4416x1242) @ 15fps
    # init_params.camera_resolution = sl.RESOLUTION.HD1080 # Use HD1080 video mode (1920x1080) @ 15fps
    init_params.camera_fps = 15 # Set the camera to 15fps, I think this is redundant as 15fps is max for HD2K
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)
    init_params.depth_maximum_distance = 16764 # Set the maximum depth perception distance to 55 ft
    # init_params.depth_minimum_distance = 13716 # Set the minimum depth perception distance to 45 ft
    # Set the positional tracking parameters
    track_params = sl.PositionalTrackingParameters()
    track_params.set_as_static = True # Set the camera to static mode
    # PositionalTrackingParameters::set_as_static 

    # Create a matrix to store image, depth, point cloud
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    # Create a reference to the mirror (IDK WHAT THIS MEANS)
    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Get the camera matrix and distortion coefficients
    cameraMatrix, distCoeffs = get_camera_matrix_and_distortion(zed)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    # A new image is available if grab() returns SUCCESS
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # Retrieve depth map. Depth is aligned on the left image
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # # choose a point in the image
        # x1, y1 = getMousePos(image.get_data())
        # print(f"Chosen point: {{{x1};{y1}}}")

        # define the image data for future use
        image_data = image.get_data()

        # detect center of the aruco marker
        x, y = find_aruco_marker(image_data, aruco_dict, parameters, cameraMatrix, distCoeffs)

        # Get the depth value at the detected point
        err, point_cloud_value = point_cloud.get_value(x, y)
        get_x_distance(point_cloud_value)

        print(f"----------------DEPTH------------------------")
        z = get_z_depth(depth, x, y)
        print(f"----------------Lateral Distance------------------------")
        x_distance = get_x_distance(point_cloud_value)
        print(f"X Distance to Target: {x_distance} mm")
        print(f"----------------YAW------------------------")
        yaw = get_camera_yaw(point_cloud_value[0], z)

        # # TESTING BEGIN
        # print(f"----------------TESTING------------------------")
        # # Print the depth value at the detected point this is the aboslute distance from the camera to the point, not the distance from the camera to the target
        # print(f"Z0 Depth value: {mm_to_feet_inches_fractions(float(depth.get_value(x, y)[1]))}, {round(depth.get_value(x, y)[1], 2)} mm")
        # x1, y1 = getMousePos(image.get_data())
        # x2, y2 = getMousePos(image.get_data())
        # x3, y3 = getMousePos(image.get_data())
        # print(f"Z1 Depth value: {mm_to_feet_inches_fractions(float(depth.get_value(x1, y1)[1]))}, {round(depth.get_value(x1, y1)[1], 2)} mm")
        # print(f"Z2 Depth value: {mm_to_feet_inches_fractions(float(depth.get_value(x2, y2)[1]))}, {round(depth.get_value(x2, y2)[1], 2)} mm")
        # print(f"Z3 Depth value: {mm_to_feet_inches_fractions(float(depth.get_value(x3, y3)[1]))}, {round(depth.get_value(x3, y3)[1], 2)} mm")
        # # TESTING END


        data = [z, x_distance, yaw]
        write_to_csv(data)

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()