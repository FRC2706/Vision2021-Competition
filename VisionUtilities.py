import math, cv2
import numpy as np
from VisionMasking import *
import datetime

# Flip image if camera mounted upside down
def flipImage(frame):
    return cv2.flip(frame, -1)


# Blurs frame
def blurImg(frame, blur_radius):
    img = frame.copy()
    blur = cv2.blur(img, (blur_radius, blur_radius))
    return blur

def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")
 
def is_cv4():
    # if we are using OpenCV 4.X, then our cv2.__version__ will start
    # with '4.'
    return check_opencv_version("4.")

def check_opencv_version(major, lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib
    # return whether or not the current OpenCV version matches the
    # major version number
    return lib.__version__.startswith(major) 

# Forgot how exactly it works, but it works! Might not be needed?
def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)


# This might be useful for something one day...?
def getEllipseRotation(image, cnt):
    try:
        # Gets rotated bounding ellipse of contour
        ellipse = cv2.fitEllipse(cnt)
        centerE = ellipse[0]
        # Gets rotation of ellipse; same as rotation of contour
        rotation = ellipse[2]
        # Gets width and height of rotated ellipse
        widthE = ellipse[1][0]
        heightE = ellipse[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, widthE, heightE)

        cv2.ellipse(image, ellipse, (23, 184, 80), 3)
        return rotation
    except:
        # Gets rotated bounding rectangle of contour
        rect = cv2.minAreaRect(cnt)
        # Creates box around that rectangle
        box = cv2.boxPoints(rect)
        # Not exactly sure
        box = np.int0(box)
        # Gets center of rotated rectangle
        center = rect[0]
        # Gets rotation of rectangle; same as rotation of contour
        rotation = rect[2]
        # Gets width and height of rotated rectangle
        width = rect[1][0]
        height = rect[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, width, height)
        return rotation   

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        green = np.uint8([[[img[y, x, 0], img[y, x, 1], img[y, x, 2]]]])
        #print(img[y, x, 2], img[y, x, 1], img[y, x, 0], cv2.cvtColor(green,cv2.COLOR_BGR2HSV))  

def milliSince1970():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970,1,1)).total_seconds()*1000)               