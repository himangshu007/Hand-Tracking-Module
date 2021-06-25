# write in terminal: pip install mediapipe
# write in terminal: pip install opencv-python

import cv2 
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLine = mpHands.HAND_CONNECTIONS

prevTime = 0
currTime = 0

while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    # to check if we hand multiple hands
    if results.multi_hand_landmarks:
        for eachHand in results.multi_hand_landmarks:
            for id , lm in enumerate(eachHand.landmark):
                # print(id , lm)
                h , w , c = img.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                print(id ,cx,cy)
                if id==0:
                    cv2.circle(img , (cx,cy) , 25 , (255,0,255) , cv2.FILLED)

            mpDraw.draw_landmarks(img , eachHand , handLine) 
    
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    cv2.putText( img , str(int(fps)) , (10 , 70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255,0,255),3)

    cv2.imshow("Image",img)
    if cv2.waitKey(1)== ord('q'):
        break