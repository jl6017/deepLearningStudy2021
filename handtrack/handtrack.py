
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    cv2.imshow("hand", img)
    cv2.waitKey(1)
# cv2.namedWindow('displaymywindows', cv2.WINDOW_NORMAL)
# CV_cat = cv2.imread('test.png')
# cv2.imshow('displaymywindows', CV_cat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()