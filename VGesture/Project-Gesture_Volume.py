import cv2
import numpy as np
import time
import HandsTrackingModule as htm
import math
# pycaw is a built lib src - github.com/AndreMiras/pycaw for volume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#################################
wCam, hCam = 640, 480  # setting width and height
#################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)  # to find position visit mediapipe site
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])     # tip of thumb and index finger

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        # centre of line
        cx, cy = (x1 + x2) // 2, (y1 + y1) // 2

        # highlighting the tips
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        # line joining the tips
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        # circle @ centre
        #cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # finding length of line
        length = math.hypot(x2 - x2, y2 - y1)
        #print(length)

        # hand range 50 - 300
        #vol range -63.5 - 0
        # changing hand range to volrange
        vol = np.interp(length, [50, 250], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)  # set the volume

        if length <= 5:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    # rectangular bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f': {int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow('Video', img)
    cv2.waitKey(1)
