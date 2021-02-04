import cv2
import numpy as np
import math
from VisionUtilities import * 
from VisionConstants import *
from DistanceFunctions import *

try:
    from PrintPublisher import *
except ImportError:
    from NetworkTablePublisher import *


# Draws Contours and finds center and yaw of orange ball
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findBall(contours, image, centerX, centerY, MergeVisionPipeLineTableName):
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)
    #cargo = []

    if len(contours) > 0:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:5]
        cntHeight = 0
        biggestPowerCell = []
        pairOfPowerCells = []
        for cnt in cntsSorted:
            x, y, w, h = cv2.boundingRect(cnt)

            ##print("Area of bounding rec: " + str(w*h))
            boundingRectArea = w*h

            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            ##print("Area of contour: " + str(cntArea))

            #percentage of contour in bounding rect
            boundingRectContArea = float(cntArea/boundingRectArea)
            #print("Percentage contour area in bounding rect: " + str(boundingRectContArea))
            #cntHeight = h
            #find the height of the bottom (y position of contour)
            # which is just the y value plus the height
            bottomHeight = y+h
            #aspect_ratio = float(w) / h
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)

            # Filters contours based off of size
            if (checkBall(cntArea, image_width, boundingRectContArea)):
                ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                if (len(biggestPowerCell) < 3):

                    ##### DRAWS CONTOUR######
                    # Gets rotated bounding rectangle of contour
                    #rect = cv2.minAreaRect(cnt)
                    # Creates box around that rectangle
                    #box = cv2.boxPoints(rect)
                    # Covert boxpoints to integer
                    #box = np.int0(box)
                   
                    # Draws a vertical white line passing through center of contour
                    cv2.line(image, (cx, screenHeight), (cx, 0), white)
                    # Draws a white circle at center of contour
                    cv2.circle(image, (cx, cy), 6, white)

                    # Draws the contours
                    cv2.drawContours(image, [cnt], 0, green, 2)

                    # Draws contour of bounding rectangle in red
                    cv2.rectangle(image, (x, y), (x + w, y + h), red, 1)
                   
                    # Appends important info to array
                    if [cx, cy, cnt, bottomHeight] not in biggestPowerCell:
                        biggestPowerCell.append([cx, cy, cnt, bottomHeight])
                        pairOfPowerCells.append(cnt)

        # Check if there are PowerCell seen
        if (len(biggestPowerCell) > 0):
            # copy
            tallestPowerCell = biggestPowerCell

            # pushes that it sees cargo to network tables

            finalTarget = []
            # Sorts targets based on tallest height (bottom of contour to top of screen or y position)
            tallestPowerCell.sort(key=lambda height: math.fabs(height[3]))


            # Sorts targets based on area
            pairOfPowerCells = sorted(pairOfPowerCells, key=lambda x: cv2.contourArea(x), reverse=True)[:2]
            if len(pairOfPowerCells) >= 2:
                M0 = cv2.moments(pairOfPowerCells[0])
                M1 = cv2.moments(pairOfPowerCells[1])
                if M0["m00"] != 0 and M1["m00"] != 0:
                    cx0 = int(M0["m10"] / M0["m00"])
                    cx1 = int(M1["m10"] / M1["m00"])    
                    avecxof2 = int((cx0+cx1)/2.0)

            #sorts closestPowerCell - contains center-x, center-y, contour and contour height from the
            #bounding rectangle.  The closest one has the largest bottom point
            closestPowerCell = min(tallestPowerCell, key=lambda height: (math.fabs(height[3] - centerX)))

            # extreme points
            #topmost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,1].argmin()][0])
            bottommost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,1].argmax()][0])

            # draw extreme points
            # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
            #cv2.circle(image, topmost, 6, white, -1)
            cv2.circle(image, bottommost, 6, blue, -1)
            ##print('extreme points', leftmost,rightmost,topmost,bottommost)

            #print("topmost: " + str(topmost[0]))
            #print("bottommost: " + str(bottommost[0]))
           
            #print("bottommost[1]: " + str(bottommost[1]))
            #print("screenheight: " + str(screenHeight))

            # Contour that fills up bottom seems to reside on one less than 
            # screen height.  For example, screenHeight of 480 has bottom
            # pixel as 479, probably because 0-479 = 480 pixel rows
            if (int(bottommost[1]) >= screenHeight - 1):
                # This is handing over centoid X when bottommost is in bottom row
                xCoord = closestPowerCell[0]
            else:
                # This is handing over X of bottommost point
                xCoord = bottommost[0]   

            # calculate yaw and store in fT0
            finalTarget.append(calculateYaw(xCoord, centerX, H_FOCAL_LENGTH))
            # calculate dist and store in fT1
            finalTarget.append(calculateDistWPILibRyan(closestPowerCell[3],TARGET_BALL_HEIGHT,KNOWN_BALL_PIXEL_HEIGHT,KNOWN_BALL_DISTANCE ))
            # calculate yaw from pure centroid and store in fT2
            finalTarget.append(calculateYaw(closestPowerCell[0], centerX, H_FOCAL_LENGTH))

            # calculate yaw to two largest contours for end trench condition, store in fT3
            if (len(biggestPowerCell) > 1):
                finalTarget.append(calculateYaw(avecxof2, centerX, H_FOCAL_LENGTH))

            #print("Yaw: " + str(finalTarget[0]))
            # Puts the yaw on screen
            # Draws yaw of target + line where center of target is
            finalYaw = round(finalTarget[1]*1000)/1000
            cv2.putText(image, "Yaw: " + str(finalTarget[0]), (40, 360), cv2.FONT_HERSHEY_COMPLEX, .6,
                        white)
            cv2.putText(image, "Dist: " + str(finalYaw), (40, 400), cv2.FONT_HERSHEY_COMPLEX, .6,
                        white)
            cv2.line(image, (xCoord, screenHeight), (xCoord, 0), blue, 2)

            cv2.putText(image, "cxYaw: " + str(finalTarget[2]), (450, 360), cv2.FONT_HERSHEY_COMPLEX, .6,
                        white)
            if (len(biggestPowerCell) > 1):
                cv2.putText(image, "cxYaw2: " + str(finalTarget[3]), (450, 400), cv2.FONT_HERSHEY_COMPLEX, .6,
                        white)

            # pushes power cell angle to network tables
            publishNumber(MergeVisionPipeLineTableName, "YawToPowerCell", finalTarget[0])
            publishNumber(MergeVisionPipeLineTableName, "DistanceToPowerCell", finalYaw)
            publishNumber(MergeVisionPipeLineTableName, "PowerCentroid1Yaw", finalTarget[2])
            if (len(biggestPowerCell) > 1):
                publishNumber(MergeVisionPipeLineTableName, "PowerCentroid2Yaw", finalTarget[3])
                cv2.line(image, (avecxof2, int(screenHeight*0.7)), (avecxof2, int(screenHeight*0.3)), green, 1)


        cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), white, 2)

        return image


# Finds the balls from the masked image and displays them on original stream + network tables
def findPowerCell(frame, mask, MergeVisionPipeLineTableName):
    # Finds contours
    if is_cv3():
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        image = findBall(contours, image, centerX, centerY, MergeVisionPipeLineTableName)
    # Shows the contours overlayed on the original video
    return image

# Checks if ball contours are worthy based off of contour area and (not currently) hull area
def checkBall(cntArea, image_width,boundingRectContArea):
    #this checks that the area of the contour is greater than the image width divide by 2
    #It also checks the percentage of the area of the bounding rectangle is
    #greater than 30%.  A single ball is usually 70-80% while groups of balls are usually
    #above 44% so using 30% is conservative
    #print("cntArea " + str(cntArea))
    return (cntArea > (image_width*2)) and (boundingRectContArea > 0.30)