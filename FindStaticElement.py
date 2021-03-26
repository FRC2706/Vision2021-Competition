import math
import numpy as np
import cv2
import operator

from VisionConstants import *
from VisionMasking import *
from VisionUtilities import *
from DistanceFunctions import *

from CornersVisual4 import get_four

try:
    from PrintPublisher import *
except ImportError:
    from NetworkTablePublisher import *

#-B -> the above target upside down, a W, drop upper center for four
real_world_coordinates = np.array([
    [-5.5625, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [5.5625, 0.0, 0.0],
    [-2.5, 6.125, 0.0],
    [2.5, 6.125, 0.0],
])

# F -> full size version of target B, also uses 5 points
#real_world_coordinates = np.array([ 
    #[-5.5625, 0.0, 0.0], # Upper left point
    #[0.0, 0.0, 0.0], # Upper center point
    #[5.5625, 0.0, 0.0], # Upper right point
    #[-2.5, 6.125, 0.0], # Bottom left point
    #[2.5, 6.125, 0.0] # Bottom right point
#])


# Finds the static elements from the masked image and displays them on original stream + network tables
def findStaticElements(frame, mask, StaticElementMethod, MergeVisionPipeLineTableName):

    # Taking a matrix of size 5 as the kernel 
    #kernel = np.ones((3,3), np.uint8) 
  
    # The first parameter is the original image, 
    # kernel is the matrix with which image is  
    # convolved and third parameter is the number  
    # of iterations, which will determine how much  
    # you want to erode/dilate a given image.  
    #img_erosion = cv2.erode(mask, kernel, iterations=1) 
    #mask = cv2.dilate(img_erosion, kernel, iterations=1) 
    #cv2.imshow("mask2", mask)

    # Finds contours
    # we are accomodating different versions of openCV and the different methods for corners
    if is_cv3():
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    else: #implies not cv3, either version 2 or 4
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:5]

    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape

    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5

    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        image = findDiamond(contours, frame, centerX, centerY, mask, StaticElementMethod, MergeVisionPipeLineTableName)
    else:
        image = frame.copy()

    return image

def findTvecRvec(image, outer_corners, real_world_coordinates):
    # Read Image
    #size = image.shape
 
    # Camera internals
 
    #focal_length = size[1]
    #center = (size[1]/2, size[0]/2)
    #camera_matrix = np.array(
    #                    [[H_FOCAL_LENGTH, 0, center[0]],
    #                    [0, V_FOCAL_LENGTH, center[1]],
    #                    [0, 0, 1]], dtype = "double"
    #                    )

    camera_matrix = np.array([[676.9254672222575, 0.0, 303.8922263320326], 
                              [0.0, 677.958895098853, 226.64055316186037], 
                              [0.0, 0.0, 1.0]], dtype = "double")
    
    dist_coeffs = np.array([[0.16171335604097975, -0.9962921370737408, -4.145368586842373e-05, 
                             0.0015152030328047668, 1.230483016701437]])

    #print("Camera Matrix :\n {0}".format(camera_matrix))                           
     #dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
 
    (success, rotation_vector, translation_vector) = cv2.solvePnP(real_world_coordinates, outer_corners, camera_matrix, dist_coeffs)
    #(success, rotation_vector, translation_vector) = cv2.solvePnP(real_world_coordinates, outer_corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_AP3P)    #(success, rotation_vector, translation_vector) = cv2.solvePnP(real_world_coordinates, outer_corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    #(success, rotation_vector, translation_vector) = cv2.solvePnP(real_world_coordinates, outer_corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    #(success, rotation_vector, translation_vector) = cv2.solvePnP(real_world_coordinates, outer_corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
    
    #print ("Rotation Vector:\n {0}".format(rotation_vector))
    #print ("Translation Vector:\n {0}".format(translation_vector))
    #print ('outer_corners:',outer_corners)
    return success, rotation_vector, translation_vector

#Computer the final output values, 
#angle 1 is the Yaw to the target
#distance is the distance to the target
#angle 2 is the Yaw of the Robot to the target

def compute_output_values(rvec, tvec):
    '''Compute the necessary output distance and angles'''

    # The tilt angle only affects the distance and angle1 calcs
    # This is a major impact on calculations
    tilt_angle = math.radians(26.18)
    distScaleFactor = 1.028
    angle2ScaleFactor = 1.0

    # Merge code last year
    x = tvec[0][0]
    y = tvec[1][0]
    z = tvec[2][0]

    # adjust z to allow calculation in horizontal plane
    z1 = math.sin(tilt_angle) * y + math.cos(tilt_angle) * z

    # distance in the horizontal plane between camera and target
    distance = math.sqrt(x**2 + z1**2) * distScaleFactor

    print('horiz distance:', distance)

    # horizontal angle between camera center line and target
    angle1InRad = math.atan2(x, z1)

    # not sure what this is, maybe the same
    angle1InRadTest = math.atan2(x, z)
    
    angle1 = math.degrees(angle1InRad)
    angle1Test = math.degrees(angle1InRadTest)

    print('angle1', angle1, angle1Test)

    rot, _ = cv2.Rodrigues(rvec)
    rot_inv = rot.transpose()
    pzero_world = np.matmul(rot_inv, -tvec)
    angle2InRad = math.atan2(pzero_world[0][0], pzero_world[2][0])
    angle2InDegrees = math.degrees(angle2InRad)

    #calculate RobotYawToDiamond based on Robot offset (subtract 180 degrees)
    #print('raw angle2 in degrees', angle2InDegrees)
    
    if angle2InDegrees < 0:
        angle2 = 180 + angle2InDegrees
    else:
        angle2 = -(180 - angle2InDegrees)

    angle2 = angle2 * angle2ScaleFactor

    print('angle2', angle2, '\n')

    return distance, angle1, angle2

#Simple function that displays 4 corners on an image
#A np.array() is expected as the input argument
def displaycorners(image, outer_corners):
    # draw extreme points
    # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
    if len(outer_corners) == 4: #this is methods 1 to 4 
        cv2.circle(image, (int(outer_corners[0,0]),int(outer_corners[0,1])), 6, cyan, -1)
        cv2.circle(image, (int(outer_corners[1,0]),int(outer_corners[1,1])), 6, red, -1)
        cv2.circle(image, (int(outer_corners[2,0]),int(outer_corners[2,1])), 6, white,-1)
        cv2.circle(image, (int(outer_corners[3,0]),int(outer_corners[3,1])), 6, blue, -1)
        #print('extreme points', leftmost,rightmost,topmost,bottommost)
    else: # this assumes len is 5 and method 5
        cv2.circle(image, (int(outer_corners[0,0]),int(outer_corners[0,1])), 6, cyan, -1)
        cv2.circle(image, (int(outer_corners[1,0]),int(outer_corners[1,1])), 6, blue, -1)
        cv2.circle(image, (int(outer_corners[2,0]),int(outer_corners[2,1])), 6, purple, -1)
        cv2.circle(image, (int(outer_corners[3,0]),int(outer_corners[3,1])), 6, white,-1)
        cv2.circle(image, (int(outer_corners[4,0]),int(outer_corners[4,1])), 6, red, -1)


# Draws Contours and finds center and yaw of vision diamond
# centerX is center x coordinate of image
# centerY is center y coordinate of image
# Draws Contours and finds center and yaw of vision diamond
# centerX is center x coordinate of image
# centerY is center y coordinate of image

def findDiamond(contours, image, centerX, centerY, mask, StaticElementMethod, MergeVisionPipeLineTableName):
    global blingColour

    angle1ScaleFactor = 1.0

    screenHeight, screenWidth, channels = image.shape
    # Seen vision diamonds (correct angle, adjacent to each other)
    diamonds = []
    # Constant used as minimum area for fingerprinting is equal to 60% of screenWidth. (Using 
    # a value based on screenWidth scales properly if the resolution ever changes.)
    minContourArea = 0.6 * screenWidth

    if len(contours) >= 1:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:5]
    
        cntsFiltered = []
        centroidDiamonds = []

        # First contour has largest area, so only go further if that one meets minimum area criterion
        if cntsSorted:

            for (j, cnt) in enumerate(cntsSorted):

                # Calculate Contour area
                cntArea = cv2.contourArea(cnt)
                # rotated rectangle fingerprinting
                rect = cv2.minAreaRect(cnt)
                (xr,yr),(wr,hr),ar = rect #x,y width, height, angle of rotation = rotated rect

                #to get rid of height and width switching
                if hr > wr: 
                    ar = ar + 90
                    wr, hr = [hr, wr]
                else:
                    ar = ar + 180
                if ar == 180:
                    ar = 0

                if hr == 0: continue
                cntAspectRatio = float(wr)/hr
                minAextent = float(cntArea)/(wr*hr)

                # Hull
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(cntArea)/hull_area

                cntsFiltered.append(cnt)
                #end fingerprinting

            # We will work on the filtered contour with the largest area which is the
            # first one in the list
            if (len(cntsFiltered) == 5):
                #print("Length of cntsFiltered:"+str(len(cntsFiltered)))

                for c in cntsFiltered:
                    M=cv2.moments(c)
                    if M["m00"] != 0:
                        cx=int(M['m10']/M["m00"])
                        cy=int(M['m01']/M["m00"])
                    else:
                        cx, cy = 0, 0

                    #print('centroid = ', cx,cy)
                    cv2.drawContours(image, [c], -1, (36,145,232), 2)
                    centroidDiamonds.append((cx,cy))


                #print('Original Centroid Diamond: ', centroidDiamonds)
                centroidDiamonds.sort(key = operator.itemgetter(0))
                #print('Centroid Diamonds sorted by x: ', centroidDiamonds)

                leftmost = centroidDiamonds[0]
                rightmost = centroidDiamonds[4]

                #centroidDiamonds.sort(key = operator.itemgetter(1))
                #print('Centroid DIamonds sorted by y: ', centroidDiamonds)

                leftother = centroidDiamonds[1]
                centerother = centroidDiamonds[2]
                rightother = centroidDiamonds[3]

                #Pick which Corner solving method to use
                foundCorners = True

                rw_coordinates = real_world_coordinates

                outer_corners = np.array([
                    leftmost,
                    centerother,
                    rightmost,
                    leftother,
                    rightother
                ], dtype="double") 

                if (foundCorners):
                    displaycorners(image, outer_corners)
                    #print('outer_corners:', outer_corners)
                    success, rvec, tvec = findTvecRvec(image, outer_corners, rw_coordinates) 

                    cx = int(centroidDiamonds[2][0])
                    cy = int(centroidDiamonds[2][1])

                    YawToDiamond = calculateYaw(cx, centerX, H_FOCAL_LENGTH) * angle1ScaleFactor
                                        
                    # If success then print values to screen                               
                    if success:
                        distance, angle1, angle2 = compute_output_values(rvec, tvec)
                        angle1 = angle1 * angle1ScaleFactor

                        cv2.putText(image, "Distance: " + str(round((distance/12),2)), (20, 430), cv2.FONT_HERSHEY_COMPLEX, 0.8,white)
                        #cv2.putText(image, "DiamondYaw: " + str(angle1), (20, 400), cv2.FONT_HERSHEY_COMPLEX, 0.8,white)
                        cv2.putText(image, "DiamondYaw: " + str(round(YawToDiamond, 2)), (20, 400), cv2.FONT_HERSHEY_COMPLEX, 0.8,white)
                        cv2.putText(image, "PhiAtDiamond: " + str(round(angle2, 2)), (20, 460), cv2.FONT_HERSHEY_COMPLEX, 0.8,white)
                        
                        # start with a non-existing colour
                        # color 0 is red
                        # color 1 is yellow
                        # color 2 is green
                        if (YawToDiamond >= -2 and YawToDiamond <= 2):
                            colour = green
                            #Use Bling
                            #Set Green colour
                            if (blingColour != 2):
                                publishNumber("blingTable", "green", 255)
                                publishNumber("blingTable", "blue", 0)
                                publishNumber("blingTable", "red", 0)
                                publishNumber("blingTable", "wait_ms", 0)
                                publishString("blingTable", "command","solid")
                                blingColour = 2
                        if ((YawToDiamond >= -5 and YawToDiamond < -2) or (YawToDiamond > 2 and YawToDiamond <= 5)):  
                            colour = yellow
                            
                            if (blingColour != 1):
                                publishNumber("blingTable", "red", 255)
                                publishNumber("blingTable", "green", 255)
                                publishNumber("blingTable", "blue", 0)
                                publishNumber("blingTable", "wait_ms", 0)
                                publishString("blingTable", "command", "solid")
                                blingColour = 1
                        if ((YawToDiamond < -5 or YawToDiamond > 5)):  
                            colour = red
                            if (blingColour != 0):
                                publishNumber("blingTable", "red", 255)
                                publishNumber("blingTable", "blue", 0)
                                publishNumber("blingTable", "green", 0)
                                publishNumber("blingTable", "wait_ms", 0)
                                publishString("blingTable", "command", "solid")
                                blingColour = 0

                        cv2.line(image, (cx, screenHeight), (cx, 0), colour, 2)
                        cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), white, 2)

                        #publishResults(name,value)
                        publishNumber(MergeVisionPipeLineTableName, "DistanceToDiamond", round(distance/12,2))
                        #publishNumber(MergeVisionPipeLineTableName, "YawToDiamond", angle1)
                        publishNumber(MergeVisionPipeLineTableName, "YawToDiamond", YawToDiamond)
                        publishNumber(MergeVisionPipeLineTableName, "RotationAngleToDiamondPerpendicular", round(angle2, 2))
                       
            else:
                #If Nothing is found, publish -99 and -1 to Network table
                publishNumber(MergeVisionPipeLineTableName, "DistanceToDiamond", -1)  
                publishNumber(MergeVisionPipeLineTableName, "YawToDiamond", -99)
                publishNumber(MergeVisionPipeLineTableName, "RotationAngleToDiamondPerpendicular", -99)
                publishString("blingTable","command","clear")

    else:
        #If Nothing is found, publish -99 and -1 to Network table
        publishNumber(MergeVisionPipeLineTableName, "DistanceToDiamond", -1) 
        publishNumber(MergeVisionPipeLineTableName, "YawToDiamond", -99)
        publishNumber(MergeVisionPipeLineTableName, "RotationAngleToDiamondPerpendicular", -99) 
        publishString("blingTable","command","clear")    
             
    return image

# Checks if the diamond contours are worthy 
def checkDiamondSize(cntArea, cntAspectRatio):
    #print("cntArea: " + str(cntArea))
    #print("aspect ratio: " + str(cntAspectRatio))
    #return (cntArea > image_width/3 and cntArea < MAXIMUM_TARGET_AREA and cntAspectRatio > 1.0)
    return (cntArea > image_width/3 and cntAspectRatio > 1.0)

if __name__ == "__main__":

    import sys

    # Print python version
    print('\n')
    print('Python version', sys.version, '\n')

    # Print version string
    cv2Version = '{0}'.format(cv2.__version__)
    print('OpenCV version', '{0}'.format(cv2.__version__), '\n')

    # setup image counter
    imageCounter = 0

    # create empty bgr image for the test
    #bgrTestImage = np.zeros(shape=[240, 320, 3], dtype=np.uint8)

    # draw a green diamond on the test image
    #pts = np.array([[200,60],[250,110],[200,160],[150,110]], np.int32)
    #bgrTestImage = cv2.drawContours(bgrTestImage,[pts],0,(0,255,0), -1)

    #pts = np.array([[200,80],[230,110],[200,140],[170,110]], np.int32)
    #bgrTestImage = cv2.drawContours(bgrTestImage,[pts],0,(0,0,0), -1)

    #bgrTestImage = cv2.imread('2021-irah4D-51T-16C/4A-04f-left.jpg')
    bgrTestImage = cv2.imread('2021-irah4D-51T-16C/4B-07f-center.jpg')
    #bgrTestImage = cv2.imread('2021-irah4D-51T-16C/4C-04f-left.jpg')
    #bgrTestImage = cv2.imread('2021-irah4D-51T-16C/4D-04f-left.jpg')

    # display the test image to verify it visually
    #cv2.imshow('This is the test image', bgrTestImage)

    # convert image to hsv from bgr
    hsvTestImage = cv2.cvtColor(bgrTestImage, cv2.COLOR_BGR2HSV)

    # using inrange from opencv make mask
    mskBinary = cv2.inRange(hsvTestImage,  (65, 50, 50), (95, 255, 255),)

    # display the mask to verify it visually
    #cv2.imshow('This is the mask', mskBinary)

    # generate the array of Contours
    contours, hierarchy = cv2.findContours(mskBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('Found', len(contours), 'contours in this photo!')

    # pass test image, binary mask and cotours to function to display all as is
    centerX = 320
    centerY = 240
    SEMethodToUse = 1
    TestNTToUse = "Test"
    #bgrfdOut = findDiamond(contours, bgrTestImage, centerX, centerY, mskBinary, SEMethodToUse, TestNTToUse)
    bgrfdOut = findDiamond(contours, bgrTestImage, centerX, centerY, mskBinary, SEMethodToUse, TestNTToUse)
    #ke = filterCirclesIn(bgrTestImage, mskBinary, contours, cv2Version)

    cv2.imshow('bgrfdOUt', bgrfdOut)

    ke = cv2.waitKey(0)

    indiv = contours[0]

    # print user keypress
    print ('ke = ', ke)

    # cleanup and exit
    cv2.destroyAllWindows()