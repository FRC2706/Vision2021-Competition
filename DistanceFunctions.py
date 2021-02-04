import math
from VisionConstants import *



def calculateDistance(heightOfCamera, heightOfTarget, pitch):
    heightOfTargetFromCamera = heightOfTarget - heightOfCamera

    # Uses trig and pitch to find distance to target
    '''
    d = distance
    h = height between camera and target
    a = angle = pitch
    tan a = h/d (opposite over adjacent)
    d = h / tan a
                         .
                        /|
                       / |
                      /  |h
                     /a  |
              camera -----
                       d
    '''
    divisor = math.tan(math.radians(pitch))
    distance = 0
    if (divisor != 0):
        distance = math.fabs(heightOfTargetFromCamera / divisor)

    return distance

avg = [0 for i in range(0, 1)]
#8 is number of frames to calculated average pixel height

def calculateDistWPILib(cntHeight, targetHeight,knownObjectPixelHeight,knownObjectDistance ):
    global image_height, avg

    for cnt in avg:
        if cnt == 0:
            cnt = cntHeight

    del avg[len(avg) - 1]
    avg.insert(0, cntHeight)
    PIX_HEIGHT = 0
    for cnt in avg:
        PIX_HEIGHT += cnt

    PIX_HEIGHT = PIX_HEIGHT / len(avg)

    #print (PIX_HEIGHT)



    #print(PIX_HEIGHT, avg)  # print("The contour height is: ", cntHeight)

    #TARGET_HEIGHT is actual height (for balls 7/12 7 inches)   
    #TARGET_HEIGHT = 0.583

 
    #image height is the y resolution calculated from image size
    #15.81 was the pixel height of a a ball found at a measured distance (which is 6 feet away)
    #65 is the pixel height of a scale image 6 feet away
    #KNOWN_OBJECT_PIXEL_HEIGHT = 65
    #KNOWN_OBJECT_DISTANCE = 6
    VIEWANGLE = math.atan((targetHeight * image_height) / (2 * knownObjectPixelHeight * knownObjectDistance))

    # print("after 2: ", VIEWANGLE)
    # VIEWANGLE = math.radians(68.5)
    distance = ((targetHeight * image_height) / (2 * PIX_HEIGHT * math.tan(VIEWANGLE)))

    return distance

# Uses trig and focal length of camera to find yaw.
# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return round(yaw,2)


# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculatePitch(pixelY, centerY, vFocalLength):
    pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
    # Just stopped working have to do this:
    pitch *= -1
    return round(pitch)

def calculateDistWPILibRyan(cntHeight, targetHeight,knownObjectPixelHeight,knownObjectDistance ):
    global image_height, avg

    for cnt in avg:
        if cnt == 0:
            cnt = cntHeight

    del avg[len(avg) - 1]
    avg.insert(0, cntHeight)
    PIX_HEIGHT = 0
    for cnt in avg:
        PIX_HEIGHT += cnt

    PIX_HEIGHT = PIX_HEIGHT / len(avg)

    #print (PIX_HEIGHT)



    #print(PIX_HEIGHT, avg)  # print("The contour height is: ", cntHeight)

    #TARGET_HEIGHT is actual height (for balls 7/12 7 inches)   
    #TARGET_HEIGHT = 0.583

 
    #image height is the y resolution calculated from image size
    #15.81 was the pixel height of a a ball found at a measured distance (which is 6 feet away)
    #65 is the pixel height of a scale image 6 feet away
    #KNOWN_OBJECT_PIXEL_HEIGHT = 65
    #KNOWN_OBJECT_DISTANCE = 6
    VIEWANGLE = 1.069283813

    # print("after 2: ", VIEWANGLE)
    # VIEWANGLE = math.radians(68.5)
    distance1 = 39.25/12*640/(2*PIX_HEIGHT*VIEWANGLE)
    VIEWANGLE1 = -0.0325*distance1 + 1.25
    distance = 39.25/12*640/(2*PIX_HEIGHT* VIEWANGLE1)

    return distance    