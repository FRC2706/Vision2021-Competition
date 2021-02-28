import cv2
import numpy as np
import math
from VisionUtilities import * 
from VisionConstants import *
from DistanceFunctions import *
#from networktables import NetworkTablesInstance
# from networktables.util import ntproperty

try:
    from PrintPublisher import *
except ImportError:
    from NetworkTablePublisher import *
    
# Note that findConeMarker uses findCone which uses checkCone

# Draws on the image - > contours and finds center and yaw of nearest powercell, and second nearest
# Puts on network tables -> Yaw and Distance to nearest yellow ball, Yaw to second nearest powercell
# frame is the original images, mask is a binary mask based on desired color
# centerX is center x coordinate of image
# centerY is center y coordinate of image
# MergeVisionPipeLineTableName is the Network Table destination for yaw and distance

# Finds the cones from the masked image and displays them on original stream + network tables
def findConeMarker(frame, mask, MergeVisionPipeLineTableName):
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
        image = findCone(contours, image, centerX, centerY, MergeVisionPipeLineTableName)
    # Shows the contours overlayed on the original video
    return image


# Draws Contours and finds center and yaw of orange cone
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findCone(contours, image, centerX, centerY, MergeVisionPipeLineTableName):
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)
 
    if len(contours) > 0:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:5]
        cntHeight = 0
        biggestCone = []
        pairOfCones = []
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
            if (checkCone(cntArea, image_width, boundingRectContArea)):
                ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                if (len(biggestCone) < 3):

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
                    if [cx, cy, cnt, bottomHeight] not in biggestCone:
                        biggestCone.append([cx, cy, cnt, bottomHeight])
                        pairOfCones.append(cnt)

        # Check if there are PowerCell seen
        if (len(biggestCone) > 0):
            # copy
            tallestCone = biggestCone

            # pushes that it sees cargo to network tables

            finalTarget = []
            # Sorts targets based on tallest height (bottom of contour to top of screen or y position)
            tallestCone.sort(key=lambda height: math.fabs(height[3]))
            print("len(tallestCone)=", len(tallestCone))

            # Sorts targets based on area  for end of trench situation, calculates average yaw (RL)
            pairOfCones = sorted(pairOfCones, key=lambda x: cv2.contourArea(x), reverse=True)[:2]
            if len(pairOfCones) >= 2:
                M0 = cv2.moments(pairOfCones[0])
                M1 = cv2.moments(pairOfCones[1])
                if M0["m00"] != 0 and M1["m00"] != 0:
                    cx0 = int(M0["m10"] / M0["m00"])
                    cx1 = int(M1["m10"] / M1["m00"])    
                    avecxof2 = int((cx0+cx1)/2.0)

            #sorts closestCone - contains center-x, center-y, contour and contour height from the
            #bounding rectangle.  The closest one has the largest bottom point
            
            closestConeList = sorted(tallestCone, key=lambda height: (math.fabs(height[3] - centerX)))
            closestCone = min(tallestCone, key=lambda height: (math.fabs(height[3] - centerX)))
            if len(closestConeList) > 1: 
                closestCone2 = closestConeList[1];

            # RL: Do calculations depending on whether there are one or two cones

            if len(closestConeList) >= 1:
                cone0 = closestConeList[0];
                topmost0 = tuple(cone0[2][cone0[2][:,:,1].argmin()][0])
                cv2.circle(image, topmost0, 6, white, -1)
                xCoord = topmost0[0];
                yaw = calculateYaw(xCoord, centerX, H_FOCAL_LENGTH)
                d = calculateDistWPILibRyan(cone0[3],TARGET_CONE_HEIGHT,KNOWN_CONE_PIXEL_HEIGHT,KNOWN_CONE_DISTANCE )

            if len(closestConeList) >= 2:
                cone1 = closestConeList[1];
                topmost1 = tuple(cone1[2][cone1[2][:,:,1].argmin()][0])
                cv2.circle(image, topmost1, 6, white, -1)
                xCoord1 = topmost1[0];
                yaw1 = calculateYaw(xCoord1, centerX, H_FOCAL_LENGTH)
                d1 = calculateDistWPILibRyan(cone1[3],TARGET_CONE_HEIGHT,KNOWN_CONE_PIXEL_HEIGHT,KNOWN_CONE_DISTANCE )

                yawRad = math.radians(yaw)
                yaw1Rad = math.radians(yaw1)
                wx = 0.5 * ( d*math.sin(yawRad) + d1*math.sin(yaw1Rad) )
                wy = 0.5 * ( d*math.cos(yawRad) + d1*math.cos(yaw1Rad) )
                d = math.sqrt(wx*wx + wy*wy)
                yawRad = math.atan(wx/wy)
                yaw = math.degrees(yawRad)
                xCoord = 160 + round(160 * math.tan(yawRad)/math.tan(horizontalView/2.0))
                print("xCoord=", xCoord)

            if len(closestConeList) > 0:
                finalTarget.append(yaw)
                finalTarget.append(d)
                finalTarget.append(yaw)


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

            # pushes cone angle to network tables
            publishNumber(MergeVisionPipeLineTableName, "YawToCone", finalTarget[0])
            publishNumber(MergeVisionPipeLineTableName, "DistanceToCone", finalYaw)
            publishNumber(MergeVisionPipeLineTableName, "ConeCentroid1Yaw", finalTarget[2])
            #if (len(biggestCone) > 1):
            #    publishNumber(MergeVisionPipeLineTableName, "ConeCentroid2Yaw", finalTarget[3])
            #    cv2.line(image, (avecxof2, int(screenHeight*0.7)), (avecxof2, int(screenHeight*0.3)), green, 1)


        #cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), white, 2)

        return image

# Checks if cone contours are worthy based off of contour area and (not currently) hull area
def checkCone(cntArea, image_width,boundingRectContArea):
    #this checks that the area of the contour is greater than the image width divide by 2
    #It also checks the percentage of the area of the bounding rectangle is
    #greater than 30%.  A single cone is usually 70-80% while groups of cones are usually
    #above 44% so using 30% is conservative
    #print("cntArea " + str(cntArea))
    #return (cntArea > (image_width*2)) and (boundingRectContArea > 0.30)
    #return (True)
    return (cntArea > 20)

if __name__ == "__main__":

    # the purpose of this code is to test the functions above
    # findConeMarker uses findCone which uses checkBall
    # this test does not use a real network table
    # TODO #2 get network tables working in test code

    # create empty bgr image for the test
    bgrTestImage = np.zeros(shape=[240, 320, 3], dtype=np.uint8)

    # draw a yellow rectangle on the test image
    bgrTestImage = cv2.circle(bgrTestImage,(100,100), 50, (0,255,255),-1)

    # display the test image to verify it visually
    cv2.imshow('This is the test', bgrTestImage)
    
    # convert image to hsv from bgr
    hsvTestImage = cv2.cvtColor(bgrTestImage, cv2.COLOR_BGR2HSV)

    # using inrange from opencv make mask
    mskBinary = cv2.inRange(hsvTestImage, (29,254,254), (31,255,255)) # (30, 255, 255)

    # display the mask to verify it visually
    cv2.imshow('This is the mask', mskBinary)

    # use a dummy network table for test code for now, real network tables not working
    MergeVisionPipeLineTableName = "DummyNetworkTableName"

    # use findConeMarker, which uses findCone, which uses checkBall to generate image
    bgrTestFoundBall = findConeMarker(bgrTestImage, mskBinary, MergeVisionPipeLineTableName)

    # display the visual output (nearest based on height) of findCone to verify it visually
    cv2.imshow('Test of 1 ball findCone output', bgrTestFoundBall)

    # wait for user input to close
    cv2.waitKey(0)

    # cleanup so we can do second test of two balls
    cv2.destroyAllWindows()

    # create empty bgr image for the test
    bgrTestImage = np.zeros(shape=[240, 320, 3], dtype=np.uint8)

    # draw two yellow circle on the test image
    bgrTestImage = cv2.circle(bgrTestImage,(100,100), 50, (0,255,255),-1)
    bgrTestImage = cv2.circle(bgrTestImage,(280,105), 45, (0,255,255),-1)

    # display the test image to verify it visually
    cv2.imshow('This is the test', bgrTestImage)
    
    # convert image to hsv from bgr
    hsvTestImage = cv2.cvtColor(bgrTestImage, cv2.COLOR_BGR2HSV)

    # using inrange from opencv make mask
    mskBinary = cv2.inRange(hsvTestImage, (29,254,254), (31,255,255)) # (30, 255, 255)

    # display the mask to verify it visually
    cv2.imshow('This is the mask', mskBinary)

    # use a dummy network table for test code for now, real network tables not working
    MergeVisionPipeLineTableName = "DummyNetworkTableName"

    # use findConeMarker, which uses findCone, which uses checkBall to generate image
    bgrTestFoundBall = findConeMarker(bgrTestImage, mskBinary, MergeVisionPipeLineTableName)

    # display the visual output (nearest based on height) of findCone to verify it visually
    cv2.imshow('Test of 2 ball findCone output', bgrTestFoundBall)

    # wait for user input to close
    cv2.waitKey(0)
    # cleanup and exit
    cv2.destroyAllWindows()
    
