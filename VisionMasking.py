import math, cv2
import numpy as np

def threshold_range(im, lo, hi):
    # For the lower limit using cv2.THRESH_BINARY, it will only include pixels that
    # are strictly greater than the threshold so need to subtract 1 from the lower
    # limit if we want the lower limit included. This can be important when thresholding
    # hue, and the lower limit is 0 (corresponding to red). This is not needed for
    # the upper limit using cv2.THRESH_BINARY_INV since the mask will be set to zero
    # if the pixel value is strictly greater than the threshold, so that values right
    # at the threshold will be set to 255 which is what we want
    unused, t1 = cv2.threshold(im, lo-1, 255, type=cv2.THRESH_BINARY)
    unused, t2 = cv2.threshold(im, hi, 255, type=cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(t1, t2)

# Masks the video based on a range of hsv colors
# Takes in a frame, range of color, and a blurred frame, returns a masked frame
def threshold_video(lower_color, upper_color, blur):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    h = threshold_range(h, lower_color[0], upper_color[0])
    s = threshold_range(s, lower_color[1], upper_color[1])
    v = threshold_range(v, lower_color[2], upper_color[2])
    combined_mask = cv2.bitwise_and(h, cv2.bitwise_and(s, v))
    
    #show the mask
    cv2.imshow("mask", combined_mask)

    # hold the HSV image to get only red colors
    # mask = cv2.inRange(combined, lower_color, upper_color)

    # Returns the masked imageBlurs video to smooth out image

    return combined_mask
