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

CHECKERBOARD_SIZE = (3, 3)

def find_checkerboard(image):

    ## IMAGE PROCESSING BEGIN ##
    # Define the region of interest for the checkerboard
    ## OUTSIDE HD2K 51.5 ft
    # y_start = 200
    # y_end = 600
    # x_start = 200
    # x_end = 800
    # ## BEDROOM HD2K 10 ft
    y_start = 400
    y_end = 800
    x_start = 600
    x_end = 1400
    # Convert the image to grayscale
    gray = cv2.cvtColor(image[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # HSV
    # lower = np.array([0, 0, 100], dtype="uint8")
    # upper = np.array([0, 0, 0], dtype="uint8")
    # hsv = cv2.cvtColor(image[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, lower, upper)
    # # Bitwise-AND mask and original image
    # krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    # dlt = cv2.dilate(mask, krn, iterations=5)
    # res = 255 - cv2.bitwise_and(dlt, mask)
    # res = np.uint8(res)
    # cv2.imshow("Processed Image", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
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
    

def main():
    image = cv2.