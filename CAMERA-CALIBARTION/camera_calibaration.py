import cv2 as cv
import os
import numpy as np

''' World point into image frame
OpenCV Python Camera Calibration (Intrinsic, Extrinsic, Distortion)
https://www.youtube.com/watch?v=H5qbRTikxI4

x  = PX 

P = K[R|T]

K = intrinsic parameters (i.e. focal length, px, py)
R, T: extrinsic matrix 
R = Rotational matrix
T = Translation matrix 

'''



# Checker board size
CHESS_BOARD_DIM = (9, 6)

# The size of Square in the checker board.
SQUARE_SIZE = 14  # millimeters

# termination criteria ?? 
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


calib_data_path = "../ArUco_marker/calib_data"
CHECK_DIR = os.path.isdir(calib_data_path) # Check is there calib_data_path

#make directory if dir does not exist 
if not CHECK_DIR:
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')

else:
    print(f'"{calib_data_path}" Directory already Exists.')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) 
obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)

obj_3D[:, :2] = np.mgrid[0 : CHESS_BOARD_DIM[0], 0 : CHESS_BOARD_DIM[1]].T.reshape(
    -1, 2
)
obj_3D *= SQUARE_SIZE # Change into World coordinate 
print(obj_3D)

# Arrays to store object points and image points from all the images.
obj_points_3D = []  # 3d point in real world space
img_points_2D = []  # 2d points in image plane.

# The images directory path
image_dir_path = "images" # in experiment, define full path for images 

files = os.listdir(image_dir_path) # get and make list of name of files in directory 
for file in files:
    print(file)
    imagePath = os.path.join(image_dir_path, file)
    # print(imagePath)

    image = cv.imread(imagePath)
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None) # ret: True, corners 
    if ret == True: # if corners are found 
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria) # ?
        img_points_2D.append(corners2)

        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

cv.destroyAllWindows()
# h, w = image.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
)
print("calibrated")

print("duming the data into one files using numpy ")
np.savez(
    f"{calib_data_path}/MultiMatrix",
    camMatrix=mtx,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)

print("-------------------------------------------")

print("loading data stored using numpy savez function\n \n \n")

data = np.load(f"{calib_data_path}/MultiMatrix.npz")

camMatrix = data["camMatrix"]
distCof = data["distCoef"] # Distortion coefficient 
rVector = data["rVector"] # Maybe matrix (rotation matrix)
tVector = data["tVector"] # 3d translation vector 

print("loaded calibration data successfully")
