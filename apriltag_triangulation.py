import cv2 as cv
import numpy as np
import pandas as pd

# Images
imgs = []

# Read apriltag detections from MATLAB output
detections = pd.read_csv("data/data.csv").to_numpy()

