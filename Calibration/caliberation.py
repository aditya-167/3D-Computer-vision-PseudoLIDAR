import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import os

def points(calibration_images, objp, rows=6, columns=9):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(calibration_images):
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (columns,rows), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(image, (9,6), corners, ret)
            plt.imshow(image)
            plt.show()
            cv2.waitKey(500)
    return objpoints, imgpoints

calibration = glob.glob('CaliberationImages/*.png')

r = 6
c = 9
square_size = 24 
objp = np.zeros((r*c,3), np.float32)
objp[:,:2] = np.mgrid[0:c, 0:r].T.reshape(-1,2)*square_size

img_shape = (640,480)
flags = (cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2+ cv2.CALIB_FIX_K3)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None, flags=flags)

