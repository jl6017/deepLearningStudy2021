import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.1, trackCon=0.1):
        self.mode = mode
        self.maxHands = maxHands
        # self.detectionCon = detectionCon
        # self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.trackCon, self.detectionCon)
        self.hands = self.mpHands.Hands(self.mode, self.maxHands)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmlist.append([id, cx, cy])
                if draw:
                    if id == 8 or id == 12 or id == 5: cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        return lmlist

def drawpoints(lmlist):
    points = []
    points.append([lmlist[8][1],lmlist[8][1]])

def main():
    pTime, cTime = 0, 0
    cap = cv2.VideoCapture(0)
    # cap.set(3,1280)
    # cap.set(4,720)
    detector = HandDetector()
    points = []
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[8], lmlist[5])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        if lmlist != []:
            if abs(lmlist[8][1]-lmlist[5][1]) + abs(lmlist[8][2]-lmlist[5][2]) > 300:
                points.append([lmlist[8][1],lmlist[8][2]])

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
        if len(points) > 1:
            for i in range(len(points) - 1):

                cx1 = points[i][0]
                cy1 = points[i][1]
                cx2 = points[i + 1][0]
                cy2 = points[i + 1][1]
                cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),20)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()