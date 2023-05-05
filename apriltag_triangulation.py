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

'''
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
'''


#RT matrix for Camera 1 is identity.
R1 = np.load(join("calibration/right", 'R1.npy'))
T1 = np.load(join("calibration/right", 't1.npy'))
RT1 = np.zeros([3,4])
RT1[:,:3] = R1
RT1[:,3:] = T1.reshape([3,1])
P1 = K_right @ RT1 #projection matrix for C1 (right camera)
 
#RT matrix for Camera 2 is the R and T obtained from stereo calibration.
R2 = np.load(join("calibration/left", 'R2.npy'))
T2 = np.load(join("calibration/left", 't2.npy'))
RT2 = np.zeros([3,4])
RT2[:,:3] = R2
RT2[:,3:] = T2.reshape([3,1])
P2 = K_left @ RT2 #projection matrix for C2 (left camera)

left_markers_blue = np.array([
    [1539.0, 1756.0, 2351.0],
    [2113.0, 1896.0, 1965.0]
])
left_markers_yellow = np.array([
    [3321.0, 2553.0, 3459.0],
    [2104.0, 1887.0, 2035.0]
])

right_markers_blue = np.array([
    [698.0, 1271.0, 1851.0],
    [2103.0, 1874.0, 1948.0]
])
right_markers_yellow = np.array([
    [2688.0, 2162.0, 2972.0],
    [2121.0, 1876.0, 2047.0]
])

# Triangulate
points_homogeneous_blue = cv.triangulatePoints(P1, P2, right_markers_blue, left_markers_blue) # Blue cone, all images
points_3d_blue = np.divide(points_homogeneous_blue[:3, :], -points_homogeneous_blue[3,:])

points_homogeneous_yellow = cv.triangulatePoints(P1, P2, right_markers_yellow, left_markers_yellow) # Yellow cone, all images
points_3d_yellow = np.divide(points_homogeneous_yellow[:3, :], -points_homogeneous_yellow[3,:])

# Plot extrinsic calibration
R1 = np.load(join('calibration/right', 'R1.npy'))
R2 = np.load(join('calibration/left', 'R2.npy'))

T1 = np.load(join('calibration/right', 't1.npy'))
T2 = np.load(join('calibration/left', 't2.npy'))


# Cone ground truth
# Left cone
LC1 = np.array([-102, 0, 228]).reshape([3,1])
LC2 = np.array([-94, 0, 385]).reshape([3,1])


# Right cone
RC1 = np.array([56, 0, 228]).reshape([3,1])
RC2 = np.array([62, 0, 385]).reshape([3,1])

# Define the camera coordinate system
cam_origin = np.array([0, 0, 0]) # camera origin
cam_x = np.array([1, 0, 0]) # camera x-axis
cam_y = np.array([0, 1, 0]) # camera y-axis
cam_z = np.array([0, 0, 1]) # camera z-axis

# Apply the rotation and translation to the camera coordinate system for camera 1
cam_origin1 = R1 @ cam_origin + T1
cam_x1 = R1 @ cam_x
cam_y1 = R1 @ cam_y
cam_z1 = R1 @ cam_z

# Apply the rotation and translation to the camera coordinate system for camera 2
cam_origin2 = R2 @ cam_origin + T2
cam_x2 = R2 @ cam_x
cam_y2 = R2 @ cam_y
cam_z2 = R2 @ cam_z

# Plot the camera coordinate systems
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(cam_origin1[0], cam_origin1[1], cam_origin1[2], 50*cam_x1[0], 50*cam_x1[1], 50*cam_x1[2], color='r')
ax.quiver(cam_origin1[0], cam_origin1[1], cam_origin1[2], 50*cam_y1[0], 50*cam_y1[1], 50*cam_y1[2], color='g')
ax.quiver(cam_origin1[0], cam_origin1[1], cam_origin1[2], 50*cam_z1[0], 50*cam_z1[1], 50*cam_z1[2], color='b')
ax.quiver(cam_origin2[0], cam_origin2[1], cam_origin2[2], 50*cam_x2[0], 50*cam_x2[1], 50*cam_x2[2], color='r')
ax.quiver(cam_origin2[0], cam_origin2[1], cam_origin2[2], 50*cam_y2[0], 50*cam_y2[1], 50*cam_y2[2], color='g')
ax.quiver(cam_origin2[0], cam_origin2[1], cam_origin2[2], 50*cam_z2[0], 50*cam_z2[1], 50*cam_z2[2], color='b')
ax.set_xlim([-400, 400])
ax.set_ylim([-400, 400])
ax.set_zlim([-400, 400])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Plot estimated cone coordinates
n = 2
ax.scatter(points_3d_blue[0,:n],points_3d_blue[1,:n],points_3d_blue[2,:n], label="Blue cone")
ax.scatter(points_3d_yellow[0,:n],points_3d_yellow[1,:n],points_3d_yellow[2,:n], label="Yellow cone")

# Plot ground truth cone coordinates
ax.scatter(LC1[0], LC1[1], LC1[2], label="Ground truth left cone 1 ")
ax.scatter(RC1[0], RC1[1], RC1[2], label="Ground truth right cone 1")
ax.scatter(LC2[0], LC2[1], LC2[2], label="Ground truth left cone 2")
ax.scatter(RC2[0], RC2[1], RC2[2], label="Ground truth right cone 2")

plt.legend()
plt.show()