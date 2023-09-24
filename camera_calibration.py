# References:
#https://stackoverflow.com/questions/22389896/finding-the-real-world-coordinates-of-an-image-point#:~:text=p%20%3D%20C%5BR%7CT,T%20is%20the%20translational%20matrix.
#https://stackoverflow.com/questions/28011873/find-world-space-coordinate-for-pixel-in-opencv
#https://docs.opencv.org/master/d7/d53/tutorial_py_pose.html
#https://stackoverflow.com/questions/31634940/compute-z-value-distance-to-camera-of-vertex-with-given-projection-matrix

import numpy as np
import cv2
import glob
import cv2.aruco as aruco

#Print out checkerboard from opencv to calibrate the camera

a = str(1)

# Defining the 9x6 checkerboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objpoints = []
imgpoints = []

# Read the checkerboard image and draw its corners
img = cv2.imread('image1' + '.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
if ret == True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    cv2.drawChessboardCorners(img, (9,6), corners2, ret)
    cv2.imshow('image1' + '.jpg', img)
    cv2.waitKey(1000)
cv2.destroyAllWindows()

# Perform camera calibration operation
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img = cv2.imread('image1' + '.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort the image and calculate the error
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('undistortedimage1' + '.jpg', dst)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
a = int(a) + 1
a = str(a)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# Get PnP Matrix of image to estimate its position in the real world
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
img = cv2.imread('image1.jpg')
ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
if ret == True:
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw(img,corners2,imgpts)
    cv2.imshow('undistortedimage1.jpg',img)
    cv2.imwrite('axisimage1.jpg', img)
    cv2.destroyAllWindows()

# Detect Aruco Markers
img = cv2.imread('image2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, None, None, None, None, None, None)
cv2.aruco.drawDetectedMarkers(img, corners, ids)
cv2.imshow('image2.jpg', img)
cv2.waitKey(10000)
rvecs, tvecs, _objpoints = cv2.aruco.estimatePoseSingleMarkers(corners, .05, newcameramtx, dist, None, None, None)

# Write new values to text files
newcameramtx = str(newcameramtx)
dist = str(dist)
file = open('Cameramtx.txt', 'w')
file.write(newcameramtx)
file.close()
file = open('Cameradist.txt', 'w')
file.write(dist)
file.close()