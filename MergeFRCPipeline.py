#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.

# My 2020 license: use it as much as you want. Crediting is recommended because it lets me know that I am being useful.
# Some parts of the architecture are based on 2019 code from the Screaming Chickens 3997

# This is meant to be used in conjuction with WPILib Raspberry Pi image: https://github.com/wpilibsuite/FRCVision-pi-gen
# ----------------------------------------------------------------------------

import json
import time
import sys
from threading import Thread
import random

from cscore import CameraServer, VideoSource
from networktables import NetworkTablesInstance
import cv2
import numpy as np
from networktables import NetworkTables
from networktables.util import ntproperty
import math

# Imports EVERYTHING from these files
from FindBall import *
from FindTarget import *
from VisionConstants import *
from VisionUtilities import *
from VisionMasking import *
from DistanceFunctions import *
from ControlPanel import *

print('OpenCV version is', cv2.__version__)

# import the necessary packages
import datetime

# class that runs separate thread for showing video,
class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, imgWidth, imgHeight, cameraServer, frame=None):
        self.outputStream = cameraServer.putVideo(OutputStream, imgWidth, imgHeight)
        #self.outputStream = cameraServer.putVideo("2706_out", imgWidth, imgHeight)
        #OutputStream
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            self.outputStream.putFrame(self.frame)

    def stop(self):
        self.stopped = True

    def notifyError(self, error):
        self.outputStream.notifyError(error)


# Class that runs a separate thread for reading  camera server also controlling exposure.
class WebcamVideoStream:
    def __init__(self, camera, cameraServer, frameWidth, frameHeight, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream

        # Automatically sets exposure to 0 to track tape
        self.webcam = camera
     
        # Some booleans so that we don't keep setting exposure over and over to the same value
        self.autoExpose = True
        self.prevValue = True

        self.switchBall = False
        self.switchTape = False
        
        # Make a blank image to write on
        self.img = np.zeros(shape=(frameWidth, frameHeight, 3), dtype=np.uint8)
        # Gets the video
        self.stream = cameraServer.getVideo()
        (self.timestamp, self.img) = self.stream.grabFrame(self.img)

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            global switch
            if self.stopped:
                return

            if switch == 1: #driver mode
                self.autoExpose = True
                ##print("Driver mode")
                if self.autoExpose != self.prevValue:
                    self.webcam.setExposureManual(60)
                    self.webcam.setExposureManual(39)
                    self.webcam.setExposureAuto()
                    ##print("Driver mode")
                    self.prevValue = self.autoExpose
             
            elif switch == 2: #Tape Target Mode - set manual exposure to 20
                #self.autoExpose = False
                #self.switchTape = True
                #if self.autoExpose != self.prevValue:
                if self.switchTape != True:
                    self.webcam.setExposureManual(60)
                    self.webcam.setExposureManual(ExposureTape)
                    self.switchTape = True
                    #self.prevValue = self.autoExpose

            elif switch == 3: #Power Cell Mode - set exposure to 39
                #self.autoExpose = False
                #if self.autoExpose != self.prevValue:
                if self.switchBall != True:
                    self.webcam.setExposureManual(100)
                    self.webcam.setExposureManual(ExposureBall)
                    self.switchBall = True
                    #self.prevValue = self.autoExpose

            # gets the image and timestamp from cameraserver
            (self.timestamp, self.img) = self.stream.grabFrame(self.img)

    def read(self):
        # return the frame most recently read
        return self.timestamp, self.img

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def getError(self):
        return self.stream.getError()


###################### PROCESSING OPENCV ################################

# counts frames for writing images
frameStop = 0
ImageCounter = 0

# Set Default to find the Tape target
#switch = 2

# Masks the video based on a range of hsv colors
# Takes in a frame, range of color, and a blurred frame, returns a masked frame
def threshold_video(lower_color, upper_color, blur):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = threshold_range(h, lower_color[0], upper_color[0])
    s = threshold_range(s, lower_color[1], upper_color[1])
    v = threshold_range(v, lower_color[2], upper_color[2])
    combined_mask = cv2.bitwise_and(h, cv2.bitwise_and(s,v))
    
    # hold the HSV image to get only red colors
    #mask = cv2.inRange(combined, lower_color, upper_color)

    # Returns the masked imageBlurs video to smooth out image
    global frameStop
    if frameStop == 1:
        global ImageCounter, matchNumber, matchNumberDefault
        matchNumber = networkTableMatch.getNumber("MatchNumber", 0)
        if matchNumber == 0:
            matchNumber = matchNumberDefault
        cv2.imwrite('/mnt/VisionImages/visionImg-' + str(matchNumber) + "-" + str(ImageCounter) + '_mask.png',
                    combined_mask)
    return combined_mask

#################### FRC VISION PI Image Specific #############
configFile = "/boot/frc.json"

pipelineConfig = "pipelineConfig.json"

with open(pipelineConfig) as json_file:
    data = json.load(json_file)

MergeVisionPipeLineTableName = data["networkTableName"]
TapeEnabled = data["Tape"]
PowerCellEnabled = data["PowerCell"]
OutputStream = data["OutputStream"]
ExposureTape = data["ExposureTarget"]
ExposureBall = data["ExposureBall"]

#print("ExposureT: "+ str(ExposureTape))
#print("ExposureB: " + str(ExposureBall))

if TapeEnabled:
    switch = 2

if PowerCellEnabled:
    switch = 3

#MergeVisionPublishingTable = "MergeVision"

class CameraConfig: pass

team = 2706
server = False
cameraConfigs = []

"""Report parse error."""

def parseError(str):
    print("config error in '" + configFile + "': " + str, file=sys.stderr)
    #Read single camera configuration.

def readCameraConfig(config):
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    cam.config = config

    cameraConfigs.append(cam)
    return True


"""Read configuration file."""

def readConfig():
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt") as f:
            j = json.load(f)
    except OSError as err:
        #print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    return True

"""Start running the camera."""
def startCamera(config):
    #print("Starting camera '{}' on {}".format(config.name, config.path))
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)
    camera.setConfigJson(json.dumps(config.config))
    return cs, camera

start, switched, prevCam = True, False, 0

currentCam = 0

def switchCam():
    global currentCam, webcam, cameras, streams, cameraServer, cap, image_width, image_height, prevCam
    if networkTableVisionPipeline.getNumber("Cam", 1):
        currentCam = 1
    else:
        currentCam = 0
    prevCam = currentCam
    cap.stop()
    webcam = cameras[currentCam]
    cameraServer = streams[currentCam]
    # Start thread reading camera
    cap = WebcamVideoStream(webcam, cameraServer, image_width, image_height).start()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    # Name of network table - this is how it communicates with robot. IMPORTANT
    #networkTable = NetworkTables.getTable(MergeVisionPublishingTable)
    networkTableMatch = NetworkTables.getTable("FMSInfo")
    networkTableTime = NetworkTables.getTable("SmartDashboard")
    networkTableMatchVariables = NetworkTables.getTable("VisionControl")

    networkTableBling = NetworkTables.getTable("blingTable")
    


    #Used to control MergeVisionPipeLineSettings
    networkTableVisionPipeline = NetworkTables.getTable(MergeVisionPipeLineTableName)


    if server:
        #print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        #print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)

    # start cameras
    cameras = []
    streams = []
    for cameraConfig in cameraConfigs:
        cs, cameraCapture = startCamera(cameraConfig)
        streams.append(cs)
        cameras.append(cameraCapture)
    # Get the first camera

    webcam = cameras[currentCam]
    cameraServer = streams[currentCam]
    # Start thread reading camera
    cap = WebcamVideoStream(webcam, cameraServer, image_width, image_height).start()
    # cap = cap.findTape
    # (optional) Setup a CvSource. This will send images back to the Dashboard
    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
    # Start thread outputing stream
    streamViewer = VideoShow(image_width, image_height, cameraServer, frame=img).start()

    # cap.autoExpose=True;
    tape = True

    
    networkTableMatchVariables.putBoolean("StartUp",False)
    networkTableMatchVariables.putBoolean("ShutDown",False)

    #PipeLine Table Values, Unique for Each PipeLine
    networkTableVisionPipeline.putBoolean("Driver", False)
    networkTableVisionPipeline.putBoolean("Tape", TapeEnabled)
    networkTableVisionPipeline.putBoolean("PowerCell", PowerCellEnabled)
    #networkTable.putBoolean("ControlPanel", False)
    networkTableVisionPipeline.putBoolean("WriteImages", False)
    networkTableVisionPipeline.putBoolean("SendMask", False)
    networkTableVisionPipeline.putBoolean("TopCamera", False)
    networkTableVisionPipeline.putBoolean("Cam", currentCam)
    #networkTable.putBoolean("Aligned", False)

    matchNumberDefault = random.randint(1, 1000)
    processed = 0

    # choose Method HERE !!!!!
    Method = 7 # likely not needed
    networkTableVisionPipeline.putNumber("Method", 7)

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

    #Setup variables for average framecount
    frameCount = 0
    averageTotal = 0
    averageFPS = 0

    framePSGroups = 50
    displayFPS = 3.14159265

    # start frames per second outside loop, will stop and restart every framePSGroups
    #fps = FPS().start()
    begin = milliSince1970()
    start = begin
    prev_update = start

    #Make sure Start and Stop images only publish network table values once
    startedImageWrite = False
    stoppedImageWrite = False

    #global blingColour
    blingColour = 0

    # loop forever
    while True:

        #print("bling Colour" + str(blingColour))
        #if networkTableTime.getNumber("Match Time", 1) == 0:
        #    networkTable.putBoolean("WriteImages", False)

        
        if (startedImageWrite == False and networkTableMatchVariables.getBoolean("StartUp",False)):
            startedImageWrite = True
            networkTableVisionPipeline.putBoolean("WriteImages", True)

        if (stoppedImageWrite == False and networkTableMatchVariables.getBoolean("ShutDown",False)):
            stoppedImageWrite = True
            networkTableVisionPipeline.putBoolean("WriteImages", False)     

        if networkTableVisionPipeline.getBoolean("TopCamera", False):
            currentCam = 1
        else:
            currentCam = 0

        if networkTableVisionPipeline.getNumber("Cam", currentCam) != prevCam:
            switchCam()

        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        timestamp, img = cap.read()
        if frameStop == 0:
            matchNumber = networkTableMatch.getNumber("MatchNumber", 0)
            if matchNumber == 0:
                matchNumber = matchNumberDefault
            cv2.imwrite('/mnt/VisionImages/visionImg-' + str(matchNumber) + "-" + str(ImageCounter) + '_Raw.png',
                        img)
        # Uncomment if camera is mounted upside down
        if networkTableVisionPipeline.getBoolean("TopCamera", False):
            frame = flipImage(img)
        else:
            frame = img
        # Comment out if camera is mounted upside down
    
        
        if timestamp == 0:
            # Send the output the error.
            streamViewer.notifyError(cap.getError())
            # skip the rest of the current iteration
            continue
        # Checks if you just want camera for driver (No processing), False by default

        #switch = 2

        

        #Check if Network Table value Tape is True
        if (networkTableVisionPipeline.getBoolean("Tape", True)):
            switch = 2
            Method = int(networkTableVisionPipeline.getNumber("Method", 7))
            threshold = threshold_video(lower_green, upper_green, frame)
            if (networkTableVisionPipeline.getBoolean("SendMask", False)):
                processed = threshold
            else:    
                processed = findTargets(frame, threshold, Method, MergeVisionPipeLineTableName)

                #Read RPM From Network Table
                rpm = networkTableVisionPipeline.getNumber("RPM", 0)
                cv2.putText(processed, "RPM: " + str(round(rpm,2)), (20, 340), cv2.FONT_HERSHEY_COMPLEX, 1.0,white)

        else:
            if (networkTableVisionPipeline.getBoolean("PowerCell", True)):
                # Checks if you just want to look for PowerCells
                switch = 3
                boxBlur = blurImg(frame, yellow_blur)
                threshold = threshold_video(lower_yellow, upper_yellow, boxBlur)
                if (networkTableVisionPipeline.getBoolean("SendMask", False)):
                    processed = threshold
                else:   
                    processed = findPowerCell(frame, threshold, MergeVisionPipeLineTableName)

            # elif (networkTableVisionPipeline.getBoolean("ControlPanel", True)):
            #     # Checks if you just want camera for Control Panel, by dent of everything else being false, true by default
            #     switch = 4
            #     #cap.autoExpose = True
            #     boxBlur = blurImg(frame, yellow_blur)
            #     # Need to create proper mask for control panel
            #     threshold = threshold_video(lower_yellow, upper_yellow, boxBlur)
            #     if (networkTableVisionPipeline.getBoolean("SendMask", False)):
            #         processed = threshold
            #     else:    
            #         processed = findControlPanel(frame, threshold)

        # Puts timestamp of camera on network tables
        networkTableVisionPipeline.putNumber("VideoTimestamp", timestamp)

        if (networkTableVisionPipeline.getBoolean("WriteImages", True)):
            frameStop = frameStop + 1
            if frameStop == 15 :
                matchNumber = networkTableMatch.getNumber("MatchNumber", 0)
                if matchNumber == 0:
                    matchNumber = matchNumberDefault
                cv2.imwrite('/mnt/VisionImages/visionImg-' +str(matchNumber)+"-"+ str(ImageCounter) + '_Proc.png', processed)
                frameStop = 0
                ImageCounter = ImageCounter+1
                if (ImageCounter==10000):
                    ImageCounter=0

        # end of cycle so update counter
        #fps.update()
        frameCount = frameCount+1
        update = milliSince1970()        

        processedMilli = (update-prev_update)
        averageTotal = averageTotal+(processedMilli)
        prev_update = update

        if ((frameCount%30)==0.0):
            averageFPS = (1000/((update-begin)/frameCount))

        # only update FPS in groups according to framePSGroups
        if frameCount%framePSGroups == 0.0:
            # also end of time we want to measure so stop FPS
            stop = milliSince1970()  
            displayFPS = (stop-start)/framePSGroups
            start = milliSince1970()

        # because we are timing in this file, have to add the fps to image processed 
        cv2.putText(processed, 'Grouped FPS: {:.2f}'.format(1000/displayFPS), (40, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6 ,white)
        cv2.putText(processed, 'Average FPS: {:.2f}'.format(averageFPS), (40, 80), cv2.FONT_HERSHEY_COMPLEX, 0.6 ,white)

        # networkTable.putBoolean("Driver", True)
        processed = cv2.resize(processed,(320,240),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
        streamViewer.frame = processed

        # Flushes camera values to reduce latency
        ntinst.flush()

    # end of while true
# end of main
# end of file