# ----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.

# My 2020 license: use it as much as you want. Crediting is recommended because it lets me know 
# that I am being useful.
# Some parts of pipeline are based on 2019 code created by the Screaming Chickens 3997

# This is meant to be used in conjuction with WPILib Raspberry Pi image: https://github.com/wpilibsuite/FRCVision-pi-gen
# ----------------------------------------------------------------------------

# import the necessary packages
import datetime
import json
import time
import sys
import random
import cv2
import math
import os
import sys

import numpy as np

from threading import Thread
from CornersVisual4 import get_four
from adrian_pyimage import FPS
from adrian_pyimage import WebcamVideoStream

# Imports EVERYTHING from these files
from FindBall import *
from FindCone import *
from FindTarget import *
from VisionConstants import *
from VisionUtilities import *
from VisionMasking import *
from DistanceFunctions import *
from ControlPanel import *
print()
print("--- Merge Viewer Starting ---")
print()
print("Using python version {0}".format(sys.version))
print()
print('OpenCV Version = ', cv2.__version__)
print()

###################### PROCESSING OPENCV ################################

# CHOOSE VIDEO OR FILES HERE!!!!
# boolean for video input, if true does video, if false images
useVideo = False
# integer for usb camera to use, boolean for live webcam
useWebCam = False
webCamNumber = 1

# ADJUST DESIRED TARGET BASED ON VIDEO OR FILES ABOVE !!!
Driver = False
Tape = False
PowerCell = False
ControlPanel = False
Cone = True

# counts frames for writing images
frameStop = 0
ImageCounter = 0
showAverageFPS = False

#Code to load images from a folder
def load_images_from_folder(folder):
    images = []
    imagename = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            imagename.append(filename)
    return images, imagename

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        green = np.uint8([[[img[y, x, 0], img[y, x, 1], img[y, x, 2]]]])
        print(img[y, x, 2], img[y, x, 1], img[y, x, 0], cv2.cvtColor(green,cv2.COLOR_BGR2HSV))  

# choose video to process -> Outer Target Videos
#videoname = './OuterTargetVideos/ThirdScale-01.mp4'
videoname = './OuterTargetVideos/FullScale-02.mp4'

if useVideo: # test against video
    showAverageFPS = True

elif useWebCam: #test against live camera
    showAverageFPS = True

else:  # implies images are to be read
    # Power Cell Images
    #images, imagename = load_images_from_folder("./PowerCellFullScale")
    #images, imagename = load_images_from_folder("./PowerCellFullMystery")
    #images, imagename = load_images_from_folder("./PowerCellFullRobot")

    # Outer Target Images
    #images, imagename = load_images_from_folder("./OuterTargetFullDistance")
    #images, imagename = load_images_from_folder("./OuterTargetImages")
    #images, imagename = load_images_from_folder("./OuterTargetRingTest")
    #images, imagename = load_images_from_folder("./OuterTargetLiger")

    #Cone Images
    images, imagename = load_images_from_folder("./OrangePylons")

    # finds height/width of camera frame (eg. 640 width, 480 height)
    image_height, image_width = images[0].shape[:2]
    print(image_height, image_width)

team = 2706
server = False
cameraConfigs = []

# Method 1 is based on measuring distance between leftmost and rightmost
# Method 2 is based on measuring the minimum enclosing circle
# Method 3 is based on measuring the major axis of the minimum enclsing ellipse
# Method 4 is a three point SolvePNP solution for distance (John and Jeremy)
# Method 5 is a four point SolvePNP solution for distance (John and Jeremy)
# Method 6 is a four point (version A) SolvePNP solution for distance (Robert, Rachel and Rebecca)
# Method 7 is a four point (version B) SolvePNP solution for distance (Robert, Rachel and Rebecca)
# Method 8 is a four point visual method using SolvePNP (Brian and Erik)
# Method 9 is a five point visual method using SolvePNP (Brian and Erik)
# Method 10 is a four point SolvePNP blending M7 and M8 (everybody!)

Method = 7

if useVideo and not useWebCam:
    cap = cv2.VideoCapture(videoname)

elif useWebCam:
    # src defines which camera, assume 2nd camera or src=1
    vs = WebcamVideoStream(src=webCamNumber).start()

else:
    currentImg = 0
    imgLength = len(images)

print("Hello Vision Team!")

stayInLoop = True

#Setup variables for average framecount
frameCount = 0
averageTotal = 0
averageFPS = 0

framePSGroups = 50
displayFPS = 3.14159265

# start
#fps = FPS().start()
begin = milliSince1970()
start = begin
prev_update = start
MergeVisionPipeLineTableName = "DummyNetworkTableName"

while stayInLoop or cap.isOpened():

    if useVideo and not useWebCam:
        (ret, frame) = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame, likely end of file, Exiting ...")
            stayInLoop = False
            break

    elif useWebCam:
        frame = vs.read()

    else:
        frame = images[currentImg]
        filename = imagename[currentImg]

    if Driver:
        processed = frame
    else:
        if Tape:
            threshold = threshold_video(lower_green, upper_green, frame)
            processed = findTargets(frame, threshold, Method, MergeVisionPipeLineTableName)
        else:
            if PowerCell:
                boxBlur = blurImg(frame, yellow_blur)
                threshold = threshold_video(lower_yellow, upper_yellow, boxBlur)
                processed = findPowerCell(frame, threshold, MergeVisionPipeLineTableName)
            elif ControlPanel:
                boxBlur = blurImg(frame, yellow_blur)
                threshold = threshold_video(lower_yellow, upper_yellow, frame)
                processed = findControlPanel(frame, threshold)
            elif Cone:
                boxBlur = blurImg(frame, yellow_blur)
                threshold = threshold_video(lower_orange, upper_orange, boxBlur)
                processed = findConeMarker(frame, threshold, MergeVisionPipeLineTableName)

    # end of cycle so update counter
    #fps.update()
    # in merge view also end of time we want to measure so stop FPS
    #fps.stop()
    frameCount = frameCount+1
    update = milliSince1970()

    processedMilli = (update-prev_update)
    averageTotal = averageTotal+(processedMilli)
    prev_update = update

    if ((frameCount%30)==0.0):
        averageFPS = (1000/((update-begin)/frameCount))

    if frameCount%framePSGroups == 0.0:
        # also end of time we want to measure so stop FPS
        stop = milliSince1970()  
        displayFPS = (stop-start)/framePSGroups
        start = milliSince1970()

    # because we are timing in this file, have to add the fps to image processed 
    #cv2.putText(processed, 'elapsed time: {:.2f}'.format(fps.elapsed()), (40, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6 ,white)
    #cv2.putText(processed, 'FPS: {:.7f}'.format(3.14159265), (40, 80), cv2.FONT_HERSHEY_COMPLEX, 0.6 ,white)
    #cv2.putText(processed, "frame time: " + str(int(processedMilli)) + " ms", (40, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6 ,white)
    #cv2.putText(processed, 'Instant FPS: {:.2f}'.format(1000/(processedMilli)), (40, 80), cv2.FONT_HERSHEY_COMPLEX, 0.6 ,white)
    
    if (showAverageFPS): 
        cv2.putText(processed, 'Grouped FPS: {:.2f}'.format(1000/(displayFPS)), (40, 200), cv2.FONT_HERSHEY_COMPLEX, 0.4, white)
        cv2.putText(processed, 'Average FPS: {:.2f}'.format(averageFPS), (40, 220), cv2.FONT_HERSHEY_COMPLEX, 0.4, white)
    else:
        cv2.putText(processed, 'Grouped FPS: {:.2f}'.format(1000/(displayFPS)), (40, 220), cv2.FONT_HERSHEY_COMPLEX, 0.4, white)

    cv2.imshow("raw", frame)
    cv2.setMouseCallback('raw', draw_circle)

    if useVideo or useWebCam:
        cv2.imshow('videoname', processed)

        key = cv2.waitKey(1)
        if key == 27:
            break

    else:
        cv2.imshow(filename, processed)

        # wait for user input to move or close
        key = cv2.waitKeyEx(0)

        if key == 113 or key == 27:
            stayInLoop = True
            break
        if key == 105 or key == 2490368:
            currentImg = currentImg - 1
            if currentImg < 0: 
                currentImg = imgLength - 1
        if key == 109 or key == 2621440:
            currentImg = currentImg + 1
            if currentImg > imgLength - 1:
                currentImg = 0

        #destroy old window
        cv2.destroyWindow(filename)
        filename = imagename[currentImg]

    # end while
# end if

if useVideo and not useWebCam:
    cap.release()
elif useWebCam:
    vs.stop()
else:
    pass

cv2.destroyAllWindows()