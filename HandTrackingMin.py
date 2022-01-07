import cv2
import mediapipe as mp
import time

OPENCV_VIDEOIO_DEBUG = 1
cap = cv2.VideoCapture(1)  # 0 is the webcam you're using - I had to disable laptop cam for this to work

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # use defaults
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):   # get pixel location
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)   # id is point location  ex: 0 is tip of thumb
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED) # this makes a large dot at id
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # capture frame rate and display on screen
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

