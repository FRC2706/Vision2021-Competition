# TimerByMerge.py
#
import cv2
import numpy as np
import datetime

black = (0, 0 ,0)
white = (255, 255, 255)
green = (0, 255, 0)
grey = (54, 54, 54)

font = cv2.FONT_HERSHEY_TRIPLEX

startTime = datetime.datetime.now()

# boolean for holding display, part of key logic below
booHold = False

# create initial black screen
image = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)

# initial start time on launch, shows it is working
startTime = datetime.datetime.now()
# initial current time is same, starts at zero
currentTime = startTime

# loop exited with break in if statements handling keyboard
while True:

    # blank image starts every to clear previous time
    image = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)

    #calculate elapsed time, convert to string
    elapsedTime = currentTime - startTime
    textTime = str(elapsedTime)

    # read desired elapsed out of string
    seconds = textTime[6:-4]
    nonSeconds = textTime[:6]
    tenSeconds = nonSeconds[5:]

    #print(elapsedTime, nonSeconds, tenSeconds, seconds)

    # put seconds to screen, only 0.00 to 9.99
    cv2.putText(image, seconds, (70, 500), font, 20.0, white, 24, cv2.FILLED)

    # depending upon how far beyond 9.99 it goes display tens of seconds
    if tenSeconds == '1':
        cv2.putText(image, '10', (100, 750), font, 8.0, white, 12, cv2.FILLED)
    if tenSeconds == '2':
        cv2.putText(image, '20', (450, 750), font, 8.0, white, 12, cv2.FILLED)
    if tenSeconds == '3':
        cv2.putText(image, '30', (800, 750), font, 8.0, white, 12, cv2.FILLED)
    if tenSeconds == '4':
        cv2.putText(image, '40', (1150, 750), font, 8.0, white, 12, cv2.FILLED)

    # display to user
    cv2.imshow('display', image)

    # get input from user if they type anything
    key = cv2.waitKeyEx(3)

    # here is the key pressed logic

    if key == 32: # space to stop
        stopTime = datetime.datetime.now()
        currentTime = stopTime
        booHold = True
        continue

    elif key == 115: # s to start
        startTime = datetime.datetime.now()
        currentTime = startTime
        booHold = False
        continue

    elif key == 122: # z to pause at zero
        startTime = datetime.datetime.now()
        currentTime = startTime
        booHold = True

    elif key == 113: # q to quit
        break     

    elif not booHold: # user has not asked for change, keep going
        currentTime = datetime.datetime.now()
        continue

# end while loop

cv2.destroyAllWindows()
