# write in terminal: pip install mediapipe
# write in terminal: pip install opencv-python

import cv2 
import time
import mediapipe as mp

class handDectector():
    def __init__(self , mode =False , maxHands = 2 , detectionCon=0.5 , trackCon = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode ,self.maxHands ,self.detectionCon,self.trackCon )
        self.mpDraw = mp.solutions.drawing_utils
        self.handLine = self.mpHands.HAND_CONNECTIONS

    def findHands(self , img , draw = True):
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

            # to check if we hand multiple hands
        if self.results.multi_hand_landmarks:
            for eachHand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img , eachHand , self.handLine) 
            
        return img

    def findPosition(self , img ,handNo=0 , draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id , lm in enumerate(myHand.landmark):
                # print(id , lm)
                h , w , c = img.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                # print(id ,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img , (cx,cy) , 25 , (255,0,255) , cv2.FILLED)

        return lmList

def main():
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)

    
    detector = handDectector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if len(lmList)!=0:
            print(lmList[4])

        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime

        cv2.putText( img , str(int(fps)) , (10 , 70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255,0,255),3)

        cv2.imshow("Image",img)
        if cv2.waitKey(1)== ord('q'):
            break



if __name__ == "__main__":
    main()
