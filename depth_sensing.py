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
MARKER_WIDTH = 300  # mm
k = 293 # mm
PRINT_DATA = False
WRITE_TO_CSV = True
SHOW_IMAGE = True
PROCESS_IMAGE = True
SET_MIN_DIST = False
GET_YAW = False
# Define the checkerboard dimensions
CHECKERBOARD_SIZE = (3, 3)


def getMousePos(image):
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['x'] = x
            param['y'] = y
            param['event'] = event



    y_start = 600
    y_end = 800
    x_start = 1000
    x_end = 1200
    param = {'x': -1, 'y': -1, 'event': -1}
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", onMouse, param)
    cv2.imshow("Image", image[y_start:y_end, x_start:x_end])

    while param['event'] != cv2.EVENT_LBUTTONDOWN:
        cv2.waitKey(10)
    
    cv2.destroyWindow("Image")
    print(f"Chosen point: {{{param['x'] + x_start};{param['y'] + y_start}}}")

    return param['x'] + x_start, param['y'] + y_start

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

def find_checkerboard(image):
    ## IMAGE PROCESSING BEGIN ##
    # Define the region of interest for the checkerboard
    # # OUTSIDE HD2K 51.5 ft
    # y_start = 200
    # y_end = 600
    # x_start = 200
    # x_end = 800
    # ## BEDROOM HD2K 10 ft
    y_start = 600
    y_end = 800
    x_start = 800
    x_end = 1400

    # Convert the image to grayscale
    gray = cv2.cvtColor(image[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
    #make gray 8 bit
    gray = np.uint8(gray)
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # gray = cv2.medianBlur(gray, 5)
    
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find the checkerboard corners
    # ret, corners = cv2.findChessboardCorners(gray, (3, 3), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret, corners = cv2.findChessboardCorners(gray, (3, 3))
    print(f"Checkerboard Corners Found: {ret}")
    # If the checkerboard corners are found
    if ret:
        # Refine the corner locations
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # Draw the checkerboard corners on the image
        cv2.drawChessboardCorners(gray, CHECKERBOARD_SIZE, corners, ret)

        # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cameraMatrix, distCoeffs)
        cv2.imshow("Checkerboard Corners", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Return the x and y coordinates of the center of the checkerboard
        x = corners[0][0][0] + x_start
        y = corners[0][0][1] + y_start
        # get the middle right edge of the checkerboard
        x_right_middle = corners[0][1][0] + x_start

        return x, y, x_right_middle
    else:
        return None

def find_aruco_marker(image, aruco_dict, parameters, cameraMatrix, distCoeffs):

    if PROCESS_IMAGE:
        # # OUTSIDE HD2K 51.5 ft
        # y_start = 200
        # y_end = 600
        # x_start = 200
        # x_end = 800
        


        # ## BEDROOM HD2K 10 ft wider view
        y_start = 625
        y_end = 725
        x_start = 1025
        x_end = 1125

        ## ----- IMAGE PROCESSING BEGIN ----- ##
        # Crop the image and convert it to grayscale
        gray = cv2.cvtColor(image[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
        # Apply bilateral filter to the gray image
        bilateral_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # conver to hsv
        # hsv = cv2.cvtColor(image[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2HSV)
        # lower_white = np.array([0, 0, 0], dtype=np.uint8)
        # upper_white = np.array([0, 0, 255], dtype=np.uint8)
        # lower_black = np.array([0, 0, 0], dtype=np.uint8)
        # upper_black = np.array([180, 255, 30], dtype=np.uint8)
        # mask_white = cv2.inRange(hsv, lower_white, upper_white)
        # mask_black = cv2.inRange(hsv, lower_black, upper_black)
        # mask = cv2.bitwise_or(mask_white, mask_black)
        # res = cv2.bitwise_and(image[y_start:y_end, x_start:x_end], image[y_start:y_end, x_start:x_end], mask=mask)
        # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        
        # Apply adaptive thresholding to segment the black and white aruco marker
        # adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # gray = cv2.medianBlur(gray, 5)
        # adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
        processed_image = gray
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
        marker_right_middle = np.mean(corners[0][0][1:3], axis=0)
        print(f"Center of the aruco marker: {marker_center}")
        print(f"Right Middle of the aruco marker: {marker_right_middle}")
        x = int(marker_center[0]) + x_start # add on the x_start to get the coordinates in the uncropped image
        y = int(marker_center[1]) + y_start # add on the y_start to get the coordinates in the uncropped image
        x_right_middle = int(marker_right_middle[0]) + x_start # add on the x_start to get the coordinates in the uncropped image

        if PRINT_DATA: print(f"Center of the aruco marker: {{{x};{y}}}")
        
        # Draw a rectangle around the detected marker in the image
        cv2.rectangle(image, (x-10, y-10), (x+10, y+10), (0, 255, 0), 2)
       
        # Display the image with the detected marker
        cv2.imshow("Raw Image with Detected Center", image[y-200:y+200, x-200:x+200])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # return the coordinates of the center of the aruco marker for depth sensing, and yaw angle
        return x, y, x_right_middle
    else:
        print("---------NO ARUCO MARKER DETECTED---------")
        return None

def get_z_depth(depth, x, y):
    # Get the depth value at the detected point
    err, z = depth.get_value(x, y)
    return z

def get_x_distance(point_cloud, x, y):
    err, point_cloud_value = point_cloud.get_value(x, y)

    # get the lateral distance to the target using the point cloud value of the center of the target
    # x_distance = point_cloud_value[0] if point_cloud_value[0] <= 0 else point_cloud_value[0] - ZED2_BASELINE/2
    if err == sl.ERROR_CODE.SUCCESS: 
        x_distance = point_cloud_value[0]
        return round(x_distance, 2)
    
    return None

def get_yaw(z1, z2):
    x0 = MARKER_WIDTH / 2
    # print(f"X0: {x0}")
    # print(f"Z1: {z1}")
    # print(f"Z2: {z2}")
    # print(f"Z2-Z1: {z2-z1}")
    # print(f"Z2-Z1/x0: {(z2-z1)/x0}")
    yaw = math.asin( (z2-z1) / x0)
    yaw_degrees = np.round(math.degrees(yaw),2)
    return yaw_degrees

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
    # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
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
    init_params.depth_minimum_distance = 13716 # Set the minimum depth perception distance to 45 ft
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
        # center_x, center_y, center_x_right = find_aruco_marker(image_data, aruco_dict, parameters, cameraMatrix, distCoeffs)

        # center_x, center_y = getMousePos(image_data)
        center_x, center_y = 1079, 682
        center_x_right = center_x + 10


        # show the image with a rectangle around the detected center
        # resize the image
        # image_data = image_data[center_y-100:center_y+100, center_x-100:center_x+100]
        # cv2.rectangle(image_data, (center_x-10, center_y-10), (center_x+10, center_y+10), (0, 255, 0), 2)
        # cv2.imshow("Raw Image with Detected Center", image_data)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        

        # detect center of the checkerboard
        # center_x, center_y, center_x_right = find_checkerboard(image_data)

        print(f"----------------DEPTH------------------------")
        z1 = get_z_depth(depth, center_x, center_y)
        z2 = get_z_depth(depth, center_x_right, center_y)
        z1_inches = mm_to_feet_inches_fractions(z1)
        z2_inches = mm_to_feet_inches_fractions(z2)
        print(f"Z Depth to Target: {z1_inches}, {round(z1, 2)} mm")
        print(f"Z Depth to Target Right: {z2_inches}, {round(z2, 2)} mm")
        print(f"----------------X Distance------------------------")
        x1 = get_x_distance(point_cloud, center_x, center_y)
        x2 = get_x_distance(point_cloud, center_x_right, center_y)
        x1_inches = mm_to_feet_inches_fractions(x1)
        x2_inches = mm_to_feet_inches_fractions(x2)
        print(f"X Distance to Target: {x1_inches}, {round(x1, 2)} mm")
        print(f"X Distance to Target Right: {x2_inches}, {round(x2, 2)} mm")
        print(f"----------------YAW------------------------")
        yaw = get_yaw(z1, z2)
        print(f"Yaw: {yaw}°")

        print(f"----------------Final Values------------------------")
        z3 = k * math.cos(math.radians(yaw))
        x3 = k * math.sin(math.radians(yaw))
        Z = np.round(z1 + z3, 2)
        X = np.round(x1 - x3, 2)
        print(f"Z: {Z} mm, {mm_to_feet_inches_fractions(Z)}")
        print(f"X: {X} mm, {mm_to_feet_inches_fractions(X)}")
        print(f"Yaw: {yaw}°")
        print(f"----------------END------------------------")

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


        data = [Z, X, yaw]
        write_to_csv(data)

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()