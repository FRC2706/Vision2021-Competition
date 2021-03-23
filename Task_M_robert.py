# This is a pseudo code file for Merge Robotics, 2020, Infinite Recharge
# This is task M - > Sample Mouse Clicks by Pixel. 
# We are going to continue towards our objective of a tool for season kickoff
# This pseudo file will allow us to determine a pixel color by clicking
# on it.  The purpose is to help calibration of the color filter.
# Using web searches for python and pixel color with a mouse, create your own
# code to deliver this capacity.
# Imports!
# modules of code as required (OpenCV here)

import argparse
import cv2
from pathlib import Path
import numpy as np

ccc = []
refPt = []
zzz = 0
blueval = 0
greenval = 0
redval = 0

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    #global image, blueval, greenval, redval

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        blueval, greenval, redval = image[y,x]
        h, s, v = hsv[y,x]
        print("blueval=", blueval, " greenval=", greenval, " redval=", redval)
        print("h=", h, " s=", s, " v=", v)

# define a string variable for the path to the file
imagePath = str(Path(__file__).parent / '2021-irahConeTesting' / 'image.png')

# load input image
image = cv2.imread(imagePath)
cv2.imshow("image", image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# set up callback for the window
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    #cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break

# close all open windows
cv2.destroyAllWindows() 
