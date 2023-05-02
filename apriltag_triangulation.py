import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from os.path import join
from scipy import linalg


# Read images
I_left1 = cv.imread('data/left/L0035.jpeg')
I_left2 = cv.imread('data/left/L0036.jpeg')
I_right1 = cv.imread('data/right/R0035.jpeg')
I_right2 = cv.imread('data/right/R0036.jpeg')

# Load from individual camera calibration
K_left = np.loadtxt("calibration/left/K.txt") # Intrinsic matrix left
DC_left = np.loadtxt("calibration/left/dc.txt") # Distortion coefficient left
K_right = np.loadtxt("calibration/right/K.txt") # Intrinsic matrix right
DC_right = np.loadtxt("calibration/right/dc.txt") # Distortion coefficient right

# Read apriltag detections from MATLAB output
detections = pd.read_csv("data/detectedAprilTags.csv").to_numpy()

right_markers = np.array(detections)[:, :2]
left_markers = np.array(detections)[:, 2:]


# Plot all detections
plt.imshow(I_left1[:, :, [2, 1, 0]])
plt.scatter(right_markers[:4, 0], right_markers[:4, 1])
plt.scatter(left_markers[:4, 0], left_markers[:4, 1])
#plt.show()

plt.imshow(I_left2[:, :, [2, 1, 0]])
plt.scatter(right_markers[4:8, 0], right_markers[4:8, 1])
plt.scatter(left_markers[4:8, 0], left_markers[4:8, 1])
#plt.show()

plt.imshow(I_right1[:, :, [2, 1, 0]])
plt.scatter(right_markers[8:12, 0], right_markers[8:12, 1])
plt.scatter(left_markers[8:12, 0], left_markers[8:12, 1])
#plt.show()

plt.imshow(I_right2[:, :, [2, 1, 0]])
plt.scatter(right_markers[12:16, 0], right_markers[12:16, 1])
plt.scatter(left_markers[12:16, 0], left_markers[12:16, 1])
#plt.show()


#RT matrix for C1 is identity.
RT1 = np.load(join("calibration/right", 'R1.npy'))
P1 = K_right @ RT1 #projection matrix for C1 (right camera)
 
#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.load(join("calibration/left", 'R2.npy'))
P2 = K_left @ RT2 #projection matrix for C2 (left camera)
