import math
import numpy as np

from VisionConstants import *
from VisionMasking import *
from VisionUtilities import *
from DistanceFunctions import *

from CornersVisual4 import get_four

try:
    from PrintPublisher import *
except ImportError:
    from NetworkTablePublisher import *


# real world dimensions of the goal target
# These are the full dimensions around both strips
TARGET_STRIP_LENGTH = 19.625    # inches
TARGET_STRIP_WIDTH = 2.0        # inches
TARGET_HEIGHT = 17.0            # inches
TARGET_TOP_WIDTH = 39.25        # inches
TARGET_BOTTOM_WIDTH = TARGET_TOP_WIDTH - 2*TARGET_STRIP_LENGTH*math.cos(math.radians(60))

#This is the X position difference between the upper target length and corner point
TARGET_BOTTOM_CORNER_WIDTH = math.sqrt(math.pow(TARGET_STRIP_LENGTH,2) - math.pow(TARGET_HEIGHT,2))

# This is the bottom width between corners
TARGET_INNER_BOTTOM_WIDTH =  TARGET_BOTTOM_WIDTH - 2.0*TARGET_STRIP_WIDTH*math.cos(math.radians(60))

# real_world_coordinates = np.array([
#     [-TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0],
#     [-TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
#     [TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
#     [TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0]
# ])

real_world_coordinates = np.array([
    [-TARGET_TOP_WIDTH / 2.0, 0.0, 0.0],
    [TARGET_TOP_WIDTH / 2.0, 0.0, 0.0],
    [-TARGET_BOTTOM_WIDTH / 2.0, TARGET_HEIGHT, 0.0],
    [TARGET_BOTTOM_WIDTH / 2.0, TARGET_HEIGHT, 0.0],
])

#top_left, top_right, bottom_left, bottom_right
# real_world_coordinates = np.array([
#     [0.0, 0.0, 0.0],             # Top Left point
#     [TARGET_TOP_WIDTH, 0.0, 0.0],           # Top Right Point
#     [TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0],            # Bottom Left point
#     [TARGET_TOP_WIDTH-TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0]          # Bottom Right point
# ])

real_world_coordinates_left = np.array([
        [0.0, 0.0, 0.0],             # Top Left point
        [0.0, 0.0, 0.0],             # Top Left point
        [TARGET_TOP_WIDTH, 0.0, 0.0],           # Top Right Point
        [TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0],         # Bottom Left point
        
    ])    

real_world_coordinates_right = np.array([
        [0.0, 0.0, 0.0],             # Top Left point
        [TARGET_TOP_WIDTH, 0.0, 0.0],           # Top Right Point
        [TARGET_TOP_WIDTH, 0.0, 0.0],           # Top Right Point,    
        [TARGET_TOP_WIDTH-TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0]     # Bottom Right point
    ])        

real_world_coordinates_inner = np.array([
    [-TARGET_TOP_WIDTH / 2.0, 0.0, 0.0],
    [-TARGET_INNER_BOTTOM_WIDTH / 2.0, TARGET_HEIGHT-2.0, 0.0],
    [TARGET_INNER_BOTTOM_WIDTH / 2.0, TARGET_HEIGHT-2.0, 0.0],
    [TARGET_TOP_WIDTH / 2.0, 0.0, 0.0],
])

real_world_coordinates_inner_five = np.array([
    [-TARGET_TOP_WIDTH / 2.0, 0.0, 0.0],
    [-TARGET_INNER_BOTTOM_WIDTH / 2.0, TARGET_HEIGHT-2.0, 0.0],
    [0.0, TARGET_HEIGHT-2.0, 0.0],
    [TARGET_INNER_BOTTOM_WIDTH / 2.0, TARGET_HEIGHT-2.0, 0.0],
    [TARGET_TOP_WIDTH / 2.0, 0.0, 0.0],
])

MAXIMUM_TARGET_AREA = 4400

# Finds the tape targets from the masked image and displays them on original stream + network tales
def findTargets(frame, mask, CornerMethod, MergeVisionPipeLineTableName):

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
        if CornerMethod is 1 or CornerMethod is 2 or CornerMethod is 3:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elif CornerMethod is 4 or CornerMethod is 5:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        elif CornerMethod is 6 or CornerMethod is 7:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif CornerMethod is 8 or CornerMethod is 9 or CornerMethod is 10:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            pass
    else: #implies not cv3, either version 2 or 4
        if CornerMethod is 1 or CornerMethod is 2 or CornerMethod is 3:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elif CornerMethod is 4 or CornerMethod is 5:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        elif CornerMethod is 6 or CornerMethod is 7:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif CornerMethod is 8 or CornerMethod is 9 or CornerMethod is 10:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            pass

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

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
        image = findTape(contours, image, centerX, centerY, mask, CornerMethod, MergeVisionPipeLineTableName)
    # Shows the contours overlayed on the original video
    return image

def get_four_points(cnt):
    # Get the left, right, and bottom points
    # extreme points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    #print('extreme points', leftmost,rightmost,topmost,bottommost)

    # Calculate centroid
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    #print('centroid = ',cx,cy)
    #cv2.line(image,(cx-10,cy-10),(cx+10,cy+10),red,2)
    #cv2.line(image,(cx-10,cy+10),(cx+10,cy-10),red,2)

    # Determine if bottom point is to the left or right of target based on centroid
    bottommost_is_left = False
    if bottommost[0] < cx:
        bottommost_is_left = True
        #print("bottommost is on the left")
    else:
        bottommost_is_left = False
        #print("bottommost is on the right") 

    # Order of points in contour appears to be top, left, bottom, right

    # Run through all points in the contour, collecting points to build lines whose
    # intersection gives the fourth point.
    topmost_index = leftmost_index = bottommost_index = rightmost_index = -1
    for i in range(len(cnt)):
        point = tuple(cnt[i][0])
        if (point == topmost):
            topmost_index = i
            #print("Found topmost:", topmost, " at index ", i)
        if (point == leftmost):
            #print("Found leftmost:", leftmost, " at index ", i)
            leftmost_index = i
        if (point == bottommost):
            #print("Found bottommost:", bottommost, " at index ", i)
            bottommost_index = i
        if (point == rightmost):
            #print("Found rightmost:", rightmost, " at index ", i)
            rightmost_index = i

    if ((topmost_index == -1)   or (leftmost_index == -1) or 
        (rightmost_index == -1) or (bottommost_index == -1)    ):
        #print ("Critical point(s) not found in contour")
        return image

    # In some cases, topmost and rightmost pixel will be the same so that index of
    # rightmost pixel in contour will be zero (instead of near the end of the contour)
    # To handle this case correctly and keep the code simple, set index of rightmost
    # pixel to be the final one in the contour. (The corresponding point and the actual
    # rightmost pixel will be very close.) 
    if rightmost_index == 0:
        rightmost_index = len(cnt-1)

    if bottommost_is_left == True:
        # Get set of points after bottommost
        num_points_to_collect = max(int(0.25*(rightmost_index-leftmost_index)), 4)
        #print("num_points_to_collect=", num_points_to_collect)
        if num_points_to_collect == 0:
            #print ("num_points_to_collect=0, exiting")
            return image
        line1_points = cnt[bottommost_index:bottommost_index+num_points_to_collect+1]
        # Get set of points before rightmost
        num_points_to_collect = max(int(0.25*(bottommost_index-leftmost_index)), 4)
        if num_points_to_collect == 0:
            #print ("num_points_to_collect=0, exiting")
            return image
        #print("num_points_to_collect=", num_points_to_collect)
        line2_points = cnt[(rightmost_index-num_points_to_collect)%len(cnt):rightmost_index+1]
    else:
        # Get set of points after leftmost
        num_points_to_collect = max(int(0.25*(rightmost_index-bottommost_index)), 4)
        if num_points_to_collect == 0:
            print ("num_points_to_collect=0, exiting")
            return image
        #print("num_points_to_collect=", num_points_to_collect)
        line1_points = cnt[leftmost_index:leftmost_index+num_points_to_collect+1]
        # Get set of point before bottommost
        num_points_to_collect = max(int(0.25*(rightmost_index-leftmost_index)), 4)
        if num_points_to_collect == 0:
            #print ("num_points_to_collect=0, exiting")
            return image
        #print("num_points_to_collect=", num_points_to_collect)
        line2_points = cnt[bottommost_index-num_points_to_collect:bottommost_index+1]


    min_points_for_line_fit = 5

    #x1 = [line1_points[i][0][0] for i in range(len(line1_points))]
    #y1 = [line1_points[i][0][1] for i in range(len(line1_points))]
    #m1, b1, r_value1, p_value1, std_err1 = stats.linregress(x1,y1)
    #print("m1=", m1, " b1=", b1)

    if len(line1_points) < min_points_for_line_fit:
        #return False, np.zeros(4,1) 
        return False, 

    [v11,v21,x01,y01] = cv2.fitLine(line1_points, cv2.DIST_L2,0,0.01,0.01)
    if (v11==0):
        #print("Warning v11=0")
        v11 = 0.1
    m1 = v21/v11
    b1 = y01 - m1*x01
    #print("From fitline: m1=", m1, " b1=", b1)

    #x2 = [line2_points[i][0][0] for i in range(len(line2_points))]
    #y2 = [line2_points[i][0][1] for i in range(len(line2_points))]
    #m2, b2, r_value2, p_value2, std_err2 = stats.linregress(x2,y2)
    #print("m2=", m2, " b2=", b2)

    if len(line2_points) < min_points_for_line_fit:
        #return False, np.zeros(4,2) 
        return False, None

    [v12,v22,x02,y02] = cv2.fitLine(line2_points, cv2.DIST_L2,0,0.01,0.01)
    m2 = v22/v12
    if (v12==0):
        print("Warning v11=0")
        v12 = 0.1
    b2 = y02 - m2*x02
    #print("From fitline: m2=", m2, " b2=", b2)

    if (m1 == m2):
        #return False, np.zeros(4,1) 
        return False, None
    xint = (b2-b1)/(m1-m2)
    yint = m1*xint+b1
    #print("xint=", xint, " yint=", yint)
    int_point = tuple([int(xint), int(yint)])

    if bottommost_is_left == True:
        four_points = np.array([
                                 leftmost,
                                 rightmost,
                                 bottommost,
                                 int_point
                                ], dtype="double")
    else:
        four_points = np.array([
                                 leftmost,
                                 rightmost,
                                 int_point,
                                 bottommost
                                ], dtype="double")

    return True, four_points

# Simple method which uses 3 Extreme points to Map the real world image
def get_four_points_with3(cnt):

    # Get extreme points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    #Set up the 3 points to map to the real world coordinates
    #print("outer left image points: " + str(outer_corners_left))
    #print("outer left world points: " + str(real_world_coordinates_left2))
    #print("outer right image points: " + str(outer_corners_right))
    #print("outer right world points: " + str(real_world_coordinates_right2))

    bottomIsLeft = True
    
    #outer corners for left side
    outer_corners = np.array([leftmost, leftmost, rightmost, bottommost], dtype="double")

    #check if bottommost is closest to right or left
    if (abs(bottommost[0]-leftmost[0]) > abs(bottommost[0]-rightmost[0])):
        #print("bottom most is right")
        bottomIsLeft = False

    if (bottomIsLeft):
        return outer_corners, real_world_coordinates_left

    outer_corners = np.array([leftmost, rightmost, rightmost, bottommost], dtype="double")
    return outer_corners, real_world_coordinates_right

# Simple method to order points from left to right
def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect

 #3D Rotation estimation
 
def findTvecRvec(image, outer_corners, real_world_coordinates):
    # Read Image
    size = image.shape
 
    # Camera internals
 
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    # camera_matrix = np.array(
    #                      [[H_FOCAL_LENGTH, 0, center[0]],
    #                      [0, V_FOCAL_LENGTH, center[1]],
    #                      [0, 0, 1]], dtype = "double"
    #                      )

    dist_coeffs = np.array([[0.16171335604097975, -0.9962921370737408, -4.145368586842373e-05, 
                             0.0015152030328047668, 1.230483016701437]])

    camera_matrix = np.array([[676.9254672222575, 0.0, 303.8922263320326], 
                              [0.0, 677.958895098853, 226.64055316186037], 
                              [0.0, 0.0, 1.0]], dtype = "double")

    #print("Camera Matrix :\n {0}".format(camera_matrix))                           
 
    #dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(real_world_coordinates, outer_corners, camera_matrix, dist_coeffs)
 
    #print ("Rotation Vector:\n {0}".format(rotation_vector))
    #print ("Translation Vector:\n {0}".format(translation_vector))
    return success, rotation_vector, translation_vector


#Computer the final output values, 
#angle 1 is the Yaw to the target
#distance is the distance to the target
#angle 2 is the Yaw of the Robot to the target
def compute_output_values(rvec, tvec):
    '''Compute the necessary output distance and angles'''

    # The tilt angle only affects the distance and angle1 calcs
    # This is a major impact on calculations
    tilt_angle = math.radians(28)

    x = tvec[0][0]
    z = math.sin(tilt_angle) * tvec[1][0] + math.cos(tilt_angle) * tvec[2][0]

    # distance in the horizontal plane between camera and target
    distance = math.sqrt(x**2 + z**2)

    # horizontal angle between camera center line and target
    angleInRad = math.atan2(x, z)
    angle1 = math.degrees(angleInRad)

    rot, _ = cv2.Rodrigues(rvec)
    rot_inv = rot.transpose()
    pzero_world = np.matmul(rot_inv, -tvec)
    angle2InRad = math.atan2(pzero_world[0][0], pzero_world[2][0])
    angle2 = math.degrees(angle2InRad)

    return distance, angle1, angle2

#Simple function that displays 4 corners on an image
#A np.array() is expected as the input argument
def displaycorners(image, outer_corners):
    # draw extreme points
    # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
    if len(outer_corners) == 4: #this is methods 1 to 4 
        cv2.circle(image, (int(outer_corners[0,0]),int(outer_corners[0,1])), 6, green, -1)
        cv2.circle(image, (int(outer_corners[1,0]),int(outer_corners[1,1])), 6, red, -1)
        cv2.circle(image, (int(outer_corners[2,0]),int(outer_corners[2,1])), 6, white,-1)
        cv2.circle(image, (int(outer_corners[3,0]),int(outer_corners[3,1])), 6, blue, -1)
        #print('extreme points', leftmost,rightmost,topmost,bottommost)
    else: # this assumes len is 5 and method 5
        cv2.circle(image, (int(outer_corners[0,0]),int(outer_corners[0,1])), 6, green, -1)
        cv2.circle(image, (int(outer_corners[1,0]),int(outer_corners[1,1])), 6, blue, -1)
        cv2.circle(image, (int(outer_corners[2,0]),int(outer_corners[2,1])), 6, purple, -1)
        cv2.circle(image, (int(outer_corners[3,0]),int(outer_corners[3,1])), 6, white,-1)
        cv2.circle(image, (int(outer_corners[4,0]),int(outer_corners[4,1])), 6, red, -1)


# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image

def findTape(contours, image, centerX, centerY, mask, CornerMethod, MergeVisionPipeLineTableName):
    global blingColour
    #global warped
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)
    targets = []
    # Constant used as minimum area for fingerprinting is equal to 60% of screenWidth. (Using 
    # a value based on screenWidth scales properly if the resolution ever changes.)
    minContourArea = 0.6 * screenWidth;

    if len(contours) >= 1:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:10]
       
        cntsFiltered = []

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

                # Filter based on area
                if (cntArea < minContourArea): continue 
                # Filter based on minimum area extent (previous values: 0.16-0.26)
                if (minAextent < 0.139 or minAextent > 1.1): continue
                # Filter based on aspect ratio (previous values: 2-3)
                if (cntAspectRatio < 1.7 or cntAspectRatio > 3.3): continue
                # Filter based on solidity (previous values: 0.22-0.35)
                if (solidity < 0.19 or solidity > 0.35): continue

                cntsFiltered.append(cnt)
                #end fingerprinting

            # We will work on the filtered contour with the largest area which is the
            # first one in the list
            if (len(cntsFiltered) > 0):

                cnt = cntsFiltered[0]

                rw_coordinates = real_world_coordinates

                #Pick which Corner solving method to use
                foundCorners = False

                if CornerMethod is 4:
                    rw_coordinates = real_world_coordinates
                    outer_corners, rw_coordinates = get_four_points_with3(cnt)
                    foundCorners = True

                elif CornerMethod is 6:
                    rw_coordinates = real_world_coordinates
                    foundCorners, outer_corners = get_four_points(cnt)

                elif CornerMethod is 7:
                    rw_coordinates = real_world_coordinates
                    foundCorners, outer_corners = get_four_points2(cnt,image)

                elif CornerMethod is 8:
                    rw_coordinates = real_world_coordinates_inner
                    xb, yb, wb, hb = cv2.boundingRect(cnt)
                    bounding_rect = (xb,yb,wb,hb)
                    ROI_mask = mask[yb:yb+hb, xb:xb+wb]
                    intROMHeight, intROMWidth = ROI_mask.shape[:2]
                    if is_cv3():
                        imgFindContourReturn, ROIcontours, hierarchy = cv2.findContours(ROI_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    else:
                        ROIcontours, hierarchy = cv2.findContours(ROI_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    ROISortedContours = sorted(ROIcontours, key = cv2.contourArea, reverse = True)[:1]
                    foundCorners, inner_corners = get_four(bounding_rect, intROMWidth, intROMHeight, ROISortedContours[0])
                    if foundCorners == True:
                        only_four = ((inner_corners[0]),(inner_corners[1]),(inner_corners[3]),(inner_corners[4]))
                        outer_corners = np.array(only_four)
                    else:
                        pass

                elif CornerMethod is 9:
                    rw_coordinates = real_world_coordinates_inner_five
                    xb, yb, wb, hb = cv2.boundingRect(cnt)
                    bounding_rect = (xb,yb,wb,hb)
                    ROI_mask = mask[yb:yb+hb, xb:xb+wb]
                    intROMHeight, intROMWidth = ROI_mask.shape[:2]
                    if is_cv3():
                        imgFindContourReturn, ROIcontours, hierarchy = cv2.findContours(ROI_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    else:
                        ROIcontours, hierarchy = cv2.findContours(ROI_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    ROISortedContours = sorted(ROIcontours, key = cv2.contourArea, reverse = True)[:1]
                    foundCorners, inner_bottom = get_four(bounding_rect, intROMWidth, intROMHeight, ROISortedContours[0])
                    if foundCorners == True:
                        outer_corners = np.array(inner_bottom)
                    else:
                        pass
                else:
                    pass

                if (foundCorners):
                    displaycorners(image, outer_corners)
                    success, rvec, tvec = findTvecRvec(image, outer_corners, rw_coordinates) 

                    #Calculate the Yaw
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0

                    YawToTarget = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    
                    # If success then print values to screen                               
                    if success:
                        distance, angle1, angle2 = compute_output_values(rvec, tvec)
                        #calculate RobotYawToTarget based on Robot offset (subtract 180 degrees)
                        RobotYawToTarget = 180-abs(angle2)
                        cv2.putText(image, "TargetYaw: " + str(YawToTarget), (20, 400), cv2.FONT_HERSHEY_COMPLEX, 1.0,white)
                        cv2.putText(image, "Distance: " + str(round((distance/12),2)), (20, 460), cv2.FONT_HERSHEY_COMPLEX, 1.0,white)
                        #cv2.putText(image, "RobotYawToTarget: " + str(round(RobotYawToTarget,2)), (40, 420), cv2.FONT_HERSHEY_COMPLEX, .6,white)
                        #cv2.putText(image, "SolvePnPTargetYawToCenter: " + str(round(angle1,2)), (40, 460), cv2.FONT_HERSHEY_COMPLEX, .6,white)
                        
                        #start with a non-existing colour
                        
                        # color 0 is red
                        # color 1 is yellow
                        # color 2 is green
                        if (YawToTarget >= -2 and YawToTarget <= 2):
                            colour = green
                            #Use Bling
                            #Set Green colour
                            if (blingColour != 2):
                                publishNumber("blingTable", "green",255)
                                publishNumber("blingTable", "blue", 0)
                                publishNumber("blingTable", "red", 0)
                                publishNumber("blingTable", "wait_ms",0)
                                publishString("blingTable","command","solid")
                                blingColour = 2
                        if ((YawToTarget >= -5 and YawToTarget < -2) or (YawToTarget > 2 and YawToTarget <= 5)):  
                            colour = yellow
                            
                            if (blingColour != 1):
                                publishNumber("blingTable", "red",255)
                                publishNumber("blingTable", "green",255)
                                publishNumber("blingTable", "blue",0)
                                publishNumber("blingTable", "wait_ms",0)
                                publishString("blingTable","command","solid")
                                blingColour = 1
                        if ((YawToTarget < -5 or YawToTarget > 5)):  
                            colour = red
                            if (blingColour != 0):
                                publishNumber("blingTable", "red",255)
                                publishNumber("blingTable", "blue",0)
                                publishNumber("blingTable", "green",0)
                                publishNumber("blingTable", "wait_ms",0)
                                publishString("blingTable","command","solid")
                                blingColour = 0

                        cv2.line(image, (cx, screenHeight), (cx, 0), colour, 2)
                        cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), white, 2)

                        #publishResults(name,value)
                        publishNumber(MergeVisionPipeLineTableName, "YawToTarget", YawToTarget)
                        publishNumber(MergeVisionPipeLineTableName, "DistanceToTarget", round(distance/12,2))
                        publishNumber(MergeVisionPipeLineTableName, "RobotYawToTarget", round(RobotYawToTarget,2))
                       
            else:
                #If Nothing is found, publish -99 and -1 to Network table
                publishNumber(MergeVisionPipeLineTableName, "YawToTarget", -99)
                publishNumber(MergeVisionPipeLineTableName, "DistanceToTarget", -1)  
                publishNumber(MergeVisionPipeLineTableName, "RobotYawToTarget", -99)
                publishString("blingTable","command","clear")


    else:
        #If Nothing is found, publish -99 and -1 to Network table
        publishNumber(MergeVisionPipeLineTableName, "YawToTarget", -99)
        publishNumber(MergeVisionPipeLineTableName, "DistanceToTarget", -1) 
        publishNumber(MergeVisionPipeLineTableName, "RobotYawToTarget", -99) 
        publishString("blingTable","command","clear")    
             
    #     # pushes vision target angle to network table
    return image

# Checks if the target contours are worthy 
def checkTargetSize(cntArea, cntAspectRatio):
    #print("cntArea: " + str(cntArea))
    #print("aspect ratio: " + str(cntAspectRatio))
    #return (cntArea > image_width/3 and cntArea < MAXIMUM_TARGET_AREA and cntAspectRatio > 1.0)
    return (cntArea > image_width/3 and cntAspectRatio > 1.0)

def get_four_points2(cnt, image):

    # Get the left and right extreme points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])

    # Order of extreme points in contour found to be: top, left, bottom, right
    # Determine indices of leftmost and rightmost point
    cnt_list = cnt[:,0].tolist()
    if list(leftmost) in cnt_list:
        leftmost_index = cnt_list.index(list(leftmost))
    else:
        print("get_four_points2(): Leftmost point not found in contour, exiting")
        return False, None
    if list(rightmost) in cnt_list:
        rightmost_index = cnt_list.index(list(rightmost))
    else:
        print("get_four_points2(): Rightmost point not found in contour, exiting")
    
    # In some cases, topmost and rightmost pixel will be the same so that index of
    # rightmost pixel in contour will be zero (instead of near the end of the contour)
    # To handle this case correctly and keep the code simple, set index of rightmost
    # pixel to be the final one in the contour. (The corresponding point and the actual
    # rightmost pixel will be very close.) 
    if rightmost_index == 0: rightmost_index = len(cnt-1)

    # For Line 1, get a set of points *after* leftmost extreme point on left part of contour
    num_points_to_collect = max(int(0.1*(rightmost_index-leftmost_index)), 4)
    if num_points_to_collect == 0:
        print ("get_four_points2(): num_points_to_collect=0 (left vertical line), exiting")
        return False, None
    line1_points = cnt[leftmost_index:leftmost_index+num_points_to_collect+1]

    # For Line 2, get a set of points around the middle of the bottom part of contour
    num_points_to_collect = max(int(0.15*(rightmost_index-leftmost_index)), 4)
    if num_points_to_collect == 0:
        print ("get_four_points2(): num_points_to_collect=0 (bottom line), exiting")
        return False, None
    approx_center_of_bottom = leftmost_index + int((rightmost_index - leftmost_index)/2)
    z =  int(num_points_to_collect/2)
    line2_points = cnt[approx_center_of_bottom-z:approx_center_of_bottom+z]

    # For Line 3, Get set of points *before* rightmost extreme point on right part of contour
    num_points_to_collect = max(int(0.1*(rightmost_index-leftmost_index)), 4)
    if num_points_to_collect == 0:
        print ("get_four_points2(): num_points_to_collect=0 (right vertical line), exiting")
        return False, None
    line3_points = cnt[(rightmost_index-num_points_to_collect)%len(cnt):rightmost_index+1]

    # Draw points found above to help a human understand what is going on
    for pt in line1_points:
        cv2.circle(image, tuple(pt[0]), 1, orange, -1)
    for pt in line2_points:
        cv2.circle(image, tuple(pt[0]), 1, orange, -1)
    for pt in line3_points:
        cv2.circle(image, tuple(pt[0]), 1, orange, -1)

    min_points_for_line_fit = 5

    # Line 1: Best fit line for left part of contour
    if len(line1_points) < min_points_for_line_fit:
        print("get_four_points2(): len(line1_points) < min_points_for_line_fit, exiting")
        return False, None
    [v11,v21,x01,y01] = cv2.fitLine(line1_points, cv2.DIST_L2,0,0.01,0.01)
    if (v11==0):
        print("get_four_points2(): Warning v11=0")
        v11 = 0.1
    m1 = v21/v11
    b1 = y01 - m1*x01

    # Line 2: Best fit line for bottom part of contour
    if len(line2_points) < min_points_for_line_fit:
        print("get_four_points2(): len(line2_points) < min_points_for_line_fit, exiting")
        return False, None
    [v12,v22,x02,y02] = cv2.fitLine(line2_points, cv2.DIST_L2,0,0.01,0.01)
    m2 = v22/v12
    if (v12==0):
        print("get_four_points2(): Warning v12=0")
        v12 = 0.1
    b2 = y02 - m2*x02

    # Line 3: Best fit line for right part of contour
    if len(line3_points) < min_points_for_line_fit:
        print("get_four_points2(): len(line3_points) < min_points_for_line_fit, exiting")
        return False, None
    [v13,v23,x03,y03] = cv2.fitLine(line3_points, cv2.DIST_L2,0,0.01,0.01)
    m3 = v23/v13
    if (v13==0):
        print("get_four_points2(): Warning v13=0")
        v13 = 0.1
    b3 = y03 - m3*x03

    # Intersection point for left bottom corner point is intersection of Lines 1 and 2
    if (m1 == m2):
        print("get_four_points2(): slope of Lines 1 and 2 equal, exiting") 
        return False, None
    xint_left = (b2-b1)/(m1-m2)
    yint_left = m1*xint_left+b1
    int_point_bottom_left = tuple([int(xint_left), int(yint_left)])

    # Intersection point for right bottom corner point is intersection of Lines 2 and 3
    if (m2 == m3):
        print("get_four_points2(): slope of Lines 2 and 3 equal, exiting") 
        return False, None
    xint_right = (b3-b2)/(m2-m3)
    yint_right = m2*xint_right+b2
    int_point_bottom_right = tuple([int(xint_right), int(yint_right)])

    # Left and right bottom points found by intersection above might actually lie on the contour. 
    # For improved accuracy, find points on contour closest to them

    cnt_pts = cnt[leftmost_index:rightmost_index]
    diffs = cnt_pts - int_point_bottom_left
    dist_sq = diffs[:,0,0]**2 + diffs[:,0,1]**2
    min_index = dist_sq.argmin()
    bottom_left = cnt_pts[min_index][0]

    cnt_pts = cnt[leftmost_index:rightmost_index]
    diffs = cnt_pts - int_point_bottom_right
    dist_sq = diffs[:,0,0]**2 + diffs[:,0,1]**2
    min_index = dist_sq.argmin()
    bottom_right = cnt_pts[min_index][0]
    
    four_points = np.array([
                            leftmost,
                            rightmost,
                            bottom_left,
                            bottom_right
                           ], dtype="double")

    return True, four_points

