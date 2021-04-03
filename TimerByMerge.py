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

booHold = False

#1920x1080
# create initial black screen
image = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)
#cv2.namedWindow('display')
#cv2.namedWindow('display', cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty('display', cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)

startTime = datetime.datetime.now()
currentTime = startTime

while True:

    image = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)

    elapsedTime = currentTime - startTime
    textTime = str(elapsedTime)

    seconds = textTime[6:-4]
    nonSeconds = textTime[:6]
    tenSeconds = nonSeconds[5:]

    #print(elapsedTime, nonSeconds, tenSeconds, seconds)

    cv2.putText(image, seconds, (70, 500), font, 20.0, white, 24, cv2.FILLED)

    if tenSeconds == '1':
        cv2.putText(image, '10', (100, 750), font, 8.0, white, 12, cv2.FILLED)
    if tenSeconds == '2':
        cv2.putText(image, '20', (450, 750), font, 8.0, white, 12, cv2.FILLED)
    if tenSeconds == '3':
        cv2.putText(image, '30', (800, 750), font, 8.0, white, 12, cv2.FILLED)
    if tenSeconds == '4':
        cv2.putText(image, '40', (1150, 750), font, 8.0, white, 12, cv2.FILLED)

    cv2.imshow('display', image)

    key = cv2.waitKeyEx(3)

    if booHold:
        pass

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

    elif not booHold:
        currentTime = datetime.datetime.now()
        continue

cv2.destroyAllWindows()
