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

# record opencv version for variability between versions
cv2Version = '{0}'.format(cv2.__version__) 

# Note that findConeMarker uses findCone which uses checkCone

# Draws on the image - > contours and finds center and yaw of nearest powercell, and second nearest
# Puts on network tables -> Yaw and Distance to nearest yellow ball, Yaw to second nearest powercell
# frame is the original images, mask is a binary mask based on desired color
# centerX is center x coordinate of image
# centerY is center y coordinate of image
# MergeVisionPipeLineTableName is the Network Table destination for yaw and distance

# Finds the cones from the masked image and displays them on original stream + network tables
def findConeMarker(frame, mask, MergeVisionPipeLineTableName):
    # Copies frame and stores it in image
    image = frame.copy()
    processed = findConeMarkerWithProcessed(frame, image, mask, MergeVisionPipeLineTableName)
    return processed

# Finds the cones from the masked image and displays them on the "processed" stream passed in as an argument + network tables
def findConeMarkerWithProcessed(frame, processed, mask, MergeVisionPipeLineTableName):
    # Finds contours
    if is_cv3():
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    #print("Number of contours: ", len(contours))

    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    # image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        processed = findCone(contours, processed, centerX, centerY, MergeVisionPipeLineTableName)
    # Shows the contours overlayed on the original video
    return processed


# Draws Contours and finds center and yaw of orange cone
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findCone(contours, image, centerX, centerY, MergeVisionPipeLineTableName):

    global cv2Version

    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)

    yaw1 = -99.0
    d1 = -99.0
    yaw2 = -99.0
    d2 = -99.0
    yawMid = -99.0
    dMid = -99.0
    phi = -99.0
 
    if len(contours) > 0:

        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:5]

        cntHeight = 0
        biggestCone = []
        pairOfCones = []

        for indiv, cnt in enumerate(cntsSorted):
        #for cnt in cntsSorted:

            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)

            # check if area too small, move on
            if (cntArea <= 50): continue

            # rotated rectangle fingerprinting
            rect = cv2.minAreaRect(cnt)
            rect = fixMinAreaRect(rect)
            (xr,yr),(wr,hr),ar = rect #x,y width, height, angle of rotation = rotated rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            #print(indiv, cntArea, ar)

            # use orientation to filter out some shapes
            if (ar <= -46) or (ar >= 46): continue
            
            x, y, w, h = cv2.boundingRect(cnt)

            if (w > h):
                w,h = h,w

            ##print("Area of bounding rec: " + str(w*h))
            boundingRectArea = w*h

            ##print("Area of contour: " + str(cntArea))
            #print("indiv=", indiv, " boundingRectArea=", boundingRectArea, " cntArea=", cntArea)

            #percentage of contour in bounding rect
            boundingExtent = float(cntArea/boundingRectArea)
            #print("Percentage contour area in bounding rect: " + str(boundingExtent))
            bottomHeight = y+h

            # Get moments of contour, mainly for centroid
            M = cv2.moments(cnt)

            # Filters contours based off of size
            if (checkCone(cntArea, image_width, boundingExtent)):
                ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0

                if (len(cntsSorted) > 0):

                    ##### DRAWS CONTOUR######                   
                    # Draws a vertical white line passing through center of contour
                    cv2.line(image, (cx, screenHeight), (cx, 0), white)
                    # Draws a white circle at center of contour
                    cv2.circle(image, (cx, cy), 6, white)

                    # Draws the contours
                    cv2.drawContours(image, [cnt], 0, green, 2)

                    # Draws contour of bounding rectangle in red
                    cv2.rectangle(image, (x, y), (x + w, y + h), red, 1)
                    cv2.drawContours(image, [box] ,0 ,blue ,2)

                    # Appends important info to array
                    if [cx, cy, cnt, bottomHeight] not in biggestCone:
                        biggestCone.append([cx, cy, cnt, h])
                        pairOfCones.append(cnt)

        # Check if there are PowerCell seen
        if (len(biggestCone) > 0):
            # copy
            tallestCone = biggestCone

            # Sorts targets based on tallest height (bottom of contour to top of screen or y position)
            tallestCone.sort(key=lambda height: math.fabs(height[3]), reverse=True)

            # Sorts targets based on area  for end of trench situation, calculates average yaw (RL)
            #pairOfCones = sorted(pairOfCones, key=lambda x: cv2.contourArea(x), reverse=True)[:2]
            #if len(pairOfCones) >= 2:
            #    M0 = cv2.moments(pairOfCones[0])
            #    M1 = cv2.moments(pairOfCones[1])
            #    if M0["m00"] != 0 and M1["m00"] != 0:
            #        cx0 = int(M0["m10"] / M0["m00"])
            #        cx1 = int(M1["m10"] / M1["m00"])    
            #        avecxof2 = int((cx0+cx1)/2.0)

            # Do calculations for distance and yaw of of nearest single cone (dSingle, yawSingle), 
            # distance and yaw of midpoint of nearest two cones (dMid, yawMid), and angle between line 
            # of sight of camera and line entering nearest two cones that is perpendicular to the line 
            # segment connecting the cones (phi). These calculations will only be done if there are an 
            # adequate number of cones found in the field of view.
            #
            # Note that these calculations use an x-y coordinate system that following the standard mathematical
            # coordinate system where the x-axis is horitontal and positive to the right, the y axis is
            # vertical and upwards. The original of the coordinate system is at the camera and the y axis
            # is along the line of sight of the camera. The camera is assumed to be horizontal and the
            # x and y axes are parallel to the ground. The z dimension is not used.
            # 

            if len(tallestCone) >= 1:
                cone1 = tallestCone[0]
                cv2.circle(image, (cone1[0], cone1[1]), 6, white, -1)
                xCoord1 = cone1[0]
                yaw1 = calculateYaw(xCoord1, centerX, H_FOCAL_LENGTH)
                #print("pixel height 1: ", cone1[3])
                d1 = calculateDistWPILibBall2021(cone1[3], TARGET_CONE_HEIGHT, tanVAConeDistance, screenWidth)
                #print("d1=", d1)

            if len(tallestCone) >= 2:
                cone2 = tallestCone[1]
                cv2.circle(image, (cone2[0], cone2[1]), 6, white, -1)
                xCoord2 = cone2[0]
                yaw2 = calculateYaw(xCoord2, centerX, H_FOCAL_LENGTH)
                #print("pixel height 2: ", cone2[3])
                d2 = calculateDistWPILibBall2021(cone2[3], TARGET_CONE_HEIGHT, tanVAConeDistance, screenWidth)
                #print("d2=", d2)
                #print("screenWidth=", screenWidth, " screenHeight=", screenHeight)

                yaw1Rad = math.radians(yaw1)
                yaw2Rad = math.radians(yaw2)

                # v1 and v2 are the vectors from the origin to the first and second nearest cones
                v1x = d1*math.sin(yaw1Rad)
                v1y = d1*math.cos(yaw1Rad)
                v2x = d2*math.sin(yaw2Rad)
                v2y = d2*math.cos(yaw2Rad)
                #print("v1x=", v1x)
                #print("v1y=", v1y)
                #print("v2x=", v2x)
                #print("v2y=", v2y)

                # w is the vector from the origin to the midpoint between the two cones and can be
                # found from simple vector math w = v1 + 0.5*(v2-v1) = 0.5*(v1+v2)
                wx = 0.5*(v1x + v2x)
                wy = 0.5*(v1y + v2y)

                # Can now calculate distance and yaw to the midpoint
                dMid = math.sqrt(wx*wx + wy*wy)
                yawMidRad = math.atan(wx/wy)
                yawMid = math.degrees(yawMidRad)

                # Calculate some intermediate quantities used later for phiMid
                v2mv1x = v2x - v1x
                v2mv1y = v2y - v1y
                #print("v2mv1x=", v2mv1x)
                #print("v2mv1y=", v2mv1y)
                 
                # thetaRad is the angle between the line of sight of the camera and the segment between
                # the two nearest cones and is used to compute phi below
                if (v2mv1y != 0.0):
                    thetaRad = math.atan(v2mv1x/v2mv1y)
                elif v2mv1x >= 0.0:
                    thetaRad = math.pi/2.0
                else:
                    thetaRad = -math.pi/2.0

                # phi is the angle between the line of sight of the camera and the line entering the two
                # cones according to the the way the robot would be expected to drive through the cones 
                # given its current position. The way it would drive through depends on a comparison between
                # the yaw angle of the first (i.e. nearest) cone and the angle theta of the line segment
                # connecting the two cones. 
                if thetaRad > yaw1Rad:
                    # Robot will drive around the right of the nearest cone and then through the two cones
                    phiRad = thetaRad - math.pi/2.0
                else:
                    # Robot will drive around the left of the nearest cone and then through the two cones
                    phiRad = thetaRad + math.pi/2.0
                phi = math.degrees(phiRad)

            # Sign convention for yaw and phi has changed at the last minute to be positive counterclockwise so
            # switch signs here
            if yaw1 != -99.0: 
                yaw1 = -yaw1
            if yaw2 != -99.0: 
                yaw2 = -yaw2
            if yawMid != -99.0: 
                yawMid = -yawMid
            if phi != -99.0: 
                phi = -phi

            # Print results on screen. Include results for single second nearest cone for debugging
            # purposes even though they are not written to the network tables.
            # Draw vertical blue line where center of through center of nearest target and vertical red
            # line through center of midpoint between the two cones

            yawSingleDisp = round(yaw1)
            dSingleDisp = round(d1, 1) 
            yaw2Disp = round(yaw2)   
            d2Disp = round(d2, 1)      
            yawMidDisp = round(yawMid)
            dMidDisp = round(dMid, 1)
            phiDisp = round(phi)

            if len(tallestCone) >= 1:
                cv2.line(image, (xCoord1, screenHeight), (xCoord1, 0), blue, 2)
            if len(tallestCone) >= 2:
                screenWidth_div_2 = round(screenWidth / 2)
                xCoordMid = screenWidth_div_2 + round(screenWidth_div_2 * math.tan(yawMidRad)/math.tan(horizontalView/2.0))
                cv2.line(image, (xCoordMid, screenHeight), (xCoordMid, 0), red, 2)

            cv2.putText(image, "Yaw Single: " + str(yawSingleDisp), (20, 160), cv2.FONT_HERSHEY_COMPLEX, .5, white)
            cv2.putText(image, "Dist Single: " + str(dSingleDisp), (20, 180), cv2.FONT_HERSHEY_COMPLEX, .5, white)
            cv2.putText(image, "Yaw2: " + str(yaw2Disp), (180, 160), cv2.FONT_HERSHEY_COMPLEX, .5, white)
            cv2.putText(image, "Dist2: " + str(d2Disp), (180, 180), cv2.FONT_HERSHEY_COMPLEX, .5, white)
            cv2.putText(image, "Yaw Mid: " + str(yawMidDisp), (340, 160), cv2.FONT_HERSHEY_COMPLEX, .5, white)
            cv2.putText(image, "Dist Mid: " + str(dMidDisp), (340, 180), cv2.FONT_HERSHEY_COMPLEX, .5, white)
            cv2.putText(image, "Phi: " + str(phiDisp), (340, 200), cv2.FONT_HERSHEY_COMPLEX, .5, white)

            # pushes cone angle to network tables
            publishNumber(MergeVisionPipeLineTableName, "YawToSingleCone", yaw1)
            publishNumber(MergeVisionPipeLineTableName, "DistanceSingleToCone", d1)
            publishNumber(MergeVisionPipeLineTableName, "YawToTwoConeMidpoint", yawMid)
            publishNumber(MergeVisionPipeLineTableName, "DistanceToTwoConeMidpoint", dMid)
            publishNumber(MergeVisionPipeLineTableName, "RotationAngleToTwoConePerpendicular", phi)

        return image

# Checks if cone contours are worthy based off of contour area and (not currently) hull area
def checkCone(cntArea, image_width,boundingExtent):
    #this checks that the area of the contour is greater than the image width divide by 2
    #It also checks the percentage of the area of the bounding rectangle is
    #greater than 30%.  A single cone is usually 70-80% while groups of cones are usually
    #above 44% so using 30% is conservative
    #print("cntArea " + str(cntArea))
    #return (cntArea > (image_width*2)) and (boundingExtent > 0.30)
    #return (True)
    #return (cntArea > 20)

    #print ("boundingExtent=", boundingExtent, " cntArea=", cntArea)
    #if ((boundingExtent < 0.4) or (cntArea < 35)):
    if ((boundingExtent < 0.35) or (cntArea < 70)):
        #print("Tossed")
        return False
    else:
        #print("Ok")
        return True
    #return (cntArea > 300)

if __name__ == "__main__":

    # the purpose of this code is to test the functions above
    # findConeMarker uses findCone which uses checkBall
    # this test does not use a real network table
    # TODO #2 get network tables working in test code

    # create empty bgr image for the test
    bgrTestImage = np.zeros(shape=[240, 320, 3], dtype=np.uint8)

    # draw a left yellow cone on the test image
    pts = np.array([[45,51],[55,51],[75,110],[35,115]], np.int32)
    bgrTestImage = cv2.drawContours(bgrTestImage,[pts],0,(0,255,255), -1)

    # draw a right green tape on the test image
    pts = np.array([[147,56],[163,51],[176,98],[160,102]], np.int32)
    bgrTestImage = cv2.drawContours(bgrTestImage,[pts],0,(0,255,255), -1)

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
    
