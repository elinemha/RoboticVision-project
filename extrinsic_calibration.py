import glob
import numpy as np
import cv2 as cv
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load images
left_images = glob.glob("data2/left/extrinsic_calibration/*.jpeg")
left_images.sort()
right_images = glob.glob("data2/right/extrinsic_calibration/*.JPG")
right_images.sort()
N_images = len(left_images)
image_size = None

# Load from individual camera calibration
K_left = np.loadtxt("calibration/left/K.txt") # Intrinsic matrix left
DC_left = np.loadtxt("calibration/left/dc.txt") # Distortion coefficient left
K_right = np.loadtxt("calibration/right/K.txt") # Intrinsic matrix right
DC_right = np.loadtxt("calibration/right/dc.txt") # Distortion coefficient right

# Load checkerboard corner setup
X_all = np.load(join("calibration", 'X_all.npy'))

# Calibration board setup
board_size  = (4,7) # Number of internal corners of the checkerboard (see tutorial)
square_size = (3.1, 3.1) # Real world length of the sides of the squares in cm

# Calculate 3D world coordinates of corners in the checkerboard frame
X_board = np.zeros((board_size[0]*board_size[1], 3), np.float32)
X_board[:,:2] = square_size*np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

# CV2 criteria
subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
stereo_calib_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)

# Corner placeholder lists
corner_coords_world = [] # 3D corner coordinates in the checkerboard frame
corner_coords_image_left = [] # 2D corner coordinates in the left image
corner_coords_image_right = [] # 2D corner coordinates in the right image


# Loop image pairs
for i in range(N_images):
    print(f"Processing image pair {i}")

    I_left = cv.imread(left_images[i], cv.IMREAD_GRAYSCALE)
    I_right = cv.imread(right_images[i], cv.IMREAD_GRAYSCALE)

    if not image_size:
        image_size = I_left.shape
        
    corner_coords_world.append(X_board)
    
    ok_left, corners_left = cv.findChessboardCorners(I_left, (board_size[0],board_size[1]), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if ok_left:
        corners_left = cv.cornerSubPix(I_left, corners_left, (11,11), (-1,-1), subpix_criteria)
        cv.drawChessboardCorners(I_left, board_size, corners_left, ok_left)
        cv.imwrite(f"output/left/{i}.jpg", I_left)
        corner_coords_image_left.append(corners_left)

    ok_right, corners_right = cv.findChessboardCorners(I_right, (board_size[0],board_size[1]), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if ok_right:
        corners_right = cv.cornerSubPix(I_right, corners_right, (11,11), (-1,-1), subpix_criteria)
        cv.drawChessboardCorners(I_right, board_size, corners_right, ok_right)
        cv.imwrite(f"output/right/{i}.jpg", I_right)
        corner_coords_image_right.append(corners_right)

    print(f"OK for image pair {i}")
    

# Stereo calibration
print("Starting stereo calibration!")
stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
ok, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(corner_coords_world, corner_coords_image_left, corner_coords_image_right, K_left, DC_left, K_right, DC_left, image_size, criteria=stereo_calib_criteria, flags=stereocalibration_flags)

if ok:
    print("Calibrated successfully :)")
    print(R)
    np.savetxt("test.txt", R)
    print(T)
else:
    print("Unsuccessful calibration ):")

# 3D plot of estimated camera setup
# Define the rotation and translation matrices for camera 1 (right camera)
R1 = np.eye(3) # rotation matrix
T1 = np.array([0,0,0]) # translation matrix


# Define the rotation and translation matrices for camera 2 (left camera)
R2 = R
T2 = T

np.save(join('calibration/right', 'R1.npy'), R1)
np.save(join('calibration/left', 'R2.npy'), R2)

np.save(join('calibration/right', 't1.npy'), T1)
np.save(join('calibration/left', 't2.npy'), T2)

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
ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([-100, 100])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
