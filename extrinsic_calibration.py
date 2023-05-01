import glob
import numpy as np
import cv2 as cv
from os.path import join

# Load images
left_images = glob.glob("data/left/extrinsic_calibration/*.jpeg")
left_images.sort()
right_images = glob.glob("data/right/extrinsic_calibration/*.jpeg")
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
        cv.imshow('img', I_left)
        k = cv.waitKey(500)
        corner_coords_image_left.append(corners_left)

    ok_right, corners_right = cv.findChessboardCorners(I_right, (board_size[0],board_size[1]), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if ok_right:
        corners_right = cv.cornerSubPix(I_right, corners_right, (11,11), (-1,-1), subpix_criteria)
        cv.drawChessboardCorners(I_right, board_size, corners_left, ok_right)
        cv.imshow('img', I_right)
        k = cv.waitKey(500)
        corner_coords_image_right.append(corners_right)

    print(f"OK for image pair {i}")
    

# Stereo calibration
print("Starting stereo calibration!")
stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
ok, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(corner_coords_world, corner_coords_image_left, corner_coords_image_right, K_left, DC_left, K_right, DC_left, image_size, criteria=stereo_calib_criteria, flags=stereocalibration_flags)

if ok:
    print("Calibrated successfully :)")
    print(T)
else:
    print("Unsuccessful calibration ):")
