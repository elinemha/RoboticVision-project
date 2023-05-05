import cv2 as cv
import numpy as np

sift = cv.xfeatures2d.SIFT_create()

# Read images
I_left1 = cv.imread('data2/left/IMG_1408.jpeg')
I_left2 = cv.imread('data2/left/IMG_1409.jpeg')
I_left3 = cv.imread('data2/left/IMG_1410.jpeg')
I_right1 = cv.imread('data2/right/IMG_2785_2.JPG')
I_right2 = cv.imread('data2/right/IMG_2786.JPG')
I_right3 = cv.imread('data2/right/IMG_2787.JPG')

# Detect keypoints and compute descriptors
kp1, desc1 = sift.detectAndCompute(I_left1, None)
kp2, desc2 = sift.detectAndCompute(I_right1, None)

# Create a BFMatcher object
bf = cv.BFMatcher()

# Find matches between the two sets of descriptors
matches = bf.knnMatch(desc1, desc2, k=2)

# Lowe's ratio test
good_matches = []
match_coords = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append(m)

        idx1 = m.queryIdx
        idx2 = m.trainIdx

        (x1, y1) = kp1[idx1].pt
        (x2, y2) = kp2[idx2].pt

        match_coords.append([x1, y1, x2, y2])


np.savetxt('matches.txt', np.array(match_coords))

# Draw the matches
match_img = cv.drawMatches(I_left1, kp1, I_right1, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
imS = cv.resize(match_img, (1920, 1040)) 
cv.imshow("Matches", imS)
cv.waitKey(0)