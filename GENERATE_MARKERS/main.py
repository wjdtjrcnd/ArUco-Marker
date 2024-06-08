import cv2 as cv
from cv2 import aruco
import os

# print(cv.__version__)

# type of markers specification 
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

MARKER_SIZE = 400  # pixels - size 

# Create the directory if it doesn't exist
if not os.path.exists("../ArUco_marker/GENERATE_MARKERS/markers"):
    os.makedirs("../ArUco_marker/GENERATE_MARKERS/markers")

n = 20 # number of desired markers

for id in range(n): #generate  n different markers  --type cd GENERATE_MARKERS then run in terminal

    marker_image = aruco.drawMarker(marker_dict, id, MARKER_SIZE)
    cv.imshow("img", marker_image)
    cv.imwrite(f"markers/markder_{id}.png", marker_image)
    # cv.waitKey(0) #if press number over zero, page up
    # break