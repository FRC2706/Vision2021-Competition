import math, cv2
import numpy as np

def threshold_range(im, lo, hi):
    unused, t1 = cv2.threshold(im, lo, 255, type=cv2.THRESH_BINARY)
    unused, t2 = cv2.threshold(im, hi, 255, type=cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(t1, t2)

# Masks the video based on a range of hsv colors
# Takes in a frame, range of color, and a blurred frame, returns a masked frame
def threshold_video(lower_color, upper_color, image):

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # the knoxville method
    #h, s, v = cv2.split(hsv)
    #h = threshold_range(h, lower_color[0], upper_color[0])
    #s = threshold_range(s, lower_color[1], upper_color[1])
    #v = threshold_range(v, lower_color[2], upper_color[2])
    #knoxville_mask = cv2.bitwise_and(h, cv2.bitwise_and(s,v))
    
    # the simple opencv mask
    inRange_mask = cv2.inRange(hsv, lower_color, upper_color)

    cleaner_mask = cv2.GaussianBlur(inRange_mask, (3,3), 0)

    return cleaner_mask
