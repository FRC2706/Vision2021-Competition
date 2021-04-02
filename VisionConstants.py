#THIS FILE CONSISTS OF VISION CONSTANTS, EXPECTED TO BE USED EVERY YEAR
import math
import numpy as np

# Field of View (FOV) of the microsoft camera (68.5 is camera spec)
# Lifecam 3000 from datasheet
# Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf

diagonalView = math.radians(68.5)

#print("Diagonal View:" + str(diagonalView))

# 4:3 aspect ratio
horizontalAspect = 4
verticalAspect = 3

# Reasons for using diagonal aspect is to calculate horizontal field of view.
diagonalAspect = math.hypot(horizontalAspect, verticalAspect)
# Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
horizontalView = math.atan(math.tan(diagonalView / 2) * (horizontalAspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView / 2) * (verticalAspect / diagonalAspect)) * 2

# RL: From camera calibration for iphone:
horizontalView = math.radians(84.5)
verticalView = math.radians(69.68)

# MAY CHANGE IN FUTURE YEARS! This is the aspect ratio used in 2020
#image_width = 640 # 4
#image_height = 480 # 3

# RL:
image_width = 320 # 4
image_height = 240 # 3


H_FOCAL_LENGTH = image_width / (2 * math.tan((horizontalView / 2)))
V_FOCAL_LENGTH = image_height / (2 * math.tan((verticalView / 2)))

#TARGET_HEIGHT is actual height (for balls 7/12 equal ball height in feet)   
TARGET_BALL_HEIGHT = 0.583
#for cones 6.5 inches - 6.5/12 (test)
TARGET_CONE_HEIGHT = 0.542

#image height is the y resolution calculated from image size
#15.81 was the pixel height of a a ball found at a measured distance (which is 6 feet away)
#65 is the pixel height of a scale image 6 feet away
KNOWN_BALL_PIXEL_HEIGHT = 65
KNOWN_BALL_DISTANCE = 6

KNOWN_CONE_PIXEL_HEIGHT = 22
KNOWN_CONE_DISTANCE = 6

#tanVA for ball distance calculation in DistanceFunctions.py (RL)
tanVABallDistance=0.295939
tanVAConeDistance = 0.4233782532

# Focal Length calculations: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_165
# H_FOCAL_LENGTH = image_width / (2 * math.tan((horizontalView / 2)))
# V_FOCAL_LENGTH = image_height / (2 * math.tan((verticalView / 2)))
# blurs have to be odd
green_blur = 1
orange_blur = 27
yellow_blur = 1

# define colors
purple = (165, 0, 120)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
cyan = (252, 252, 3)
white = (255, 255, 255)
yellow = (0, 255, 255)
orange = (60, 255, 255)

# define range of green of retroreflective tape in HSV
#lower_green = np.array([55, 55, 55])
#upper_green = np.array([100, 255, 255])

# define range of green of retroreflective tape in HSV
lower_green = np.array([65, 50, 50])
upper_green = np.array([95, 255, 255])


# define range of green of retroreflective tape in HSV
#lower_green = np.array([23, 50, 35])
#upper_green = np.array([85, 255, 255])

lower_yellow = np.array([10, 150, 65]) # was 14, 150, 150
upper_yellow = np.array([30, 255, 255])

#define range of orange of cone in HSV
#lower_orange = np.array([1, 130, 170])
#upper_orange = np.array([18, 255, 255])
# New range from Jamie
#lower_orange = np.array([1, 190, 130])
#upper_orange = np.array([18, 255, 255])
# Brian
#lower_orange = np.array([0, 100, 100])
#upper_orange = np.array([8, 255, 255])
# Robert experiment night dining room
#lower_orange = np.array([1, 150, 150])
#upper_orange = np.array([4, 255, 255])
# Robert experiment day dining room
#lower_orange = np.array([0, 140, 180])
#upper_orange = np.array([12, 255, 255])
# Brian picture 
#lower_orange = np.array([0, 200, 150])
#upper_orange = np.array([3, 255, 255])
# Robert experiment night 2  dining room
#lower_orange = np.array([0, 200, 175])
#upper_orange = np.array([3, 255, 255])
# Wei basement 6:20pm (no lights)
#lower_orange = np.array([0, 125, 160])
#upper_orange = np.array([6, 190, 255])
# Wei basement 6:40pm (lights on)
#lower_orange = np.array([8, 100, 150])
#upper_orange = np.array([17, 150, 200])
# 2021-irahConeTesting
#lower_orange = np.array([0, 120, 170])
#upper_orange = np.array([19, 200, 255])
# 2021-03-27 rl basement
lower_orange = np.array([10, 100, 140])
upper_orange = np.array([23, 255, 255])


blingColour = 0

# Very simple correction factor used in distance compuation function calculateDistWPILibBall2021 
# in DistanceFunctions.py. Take a measurement from a known distance and set it equal to measurement
# divided by known disance. If you don't know what to put, set this to 1.
DISTANCE_CORRECTION_FACTOR = 3.0/5.0