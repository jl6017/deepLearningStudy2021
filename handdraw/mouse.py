import numpy as np
import cv2 as cv
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import modelcnn
import modellinear
import modelresnet

idx = ["0","1","2","3","4","5","6","7","8","9",
"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
"a","b","d","e","f","g","h","n","q","r","t"]
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.circle(img,(x,y),10,(255,255,255),cv.FILLED)
            # else:
            #     cv.circle(img,(x,y),5,(0,0,255),-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        # if mode == True:
        #     cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        # else:
        #     cv.circle(img,(x,y),5,(0,0,255),-1)

def predict(img):
    newimg = cv.resize(img,(32,32))
    myimg = torch.tensor(newimg)
    # print(myimg.shape)
    x = (myimg/255).unsqueeze(0).unsqueeze(0)
    # print(x)
    y = model(x)
    _, pred = torch.max(y, dim=1)
    return idx[pred.item()] 

img = np.zeros((512,512,1), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
model = modelresnet.getmodel()
# model = modellinear.getlinear()
text = ""
while(1):
    # cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode 
    elif k == ord('a'):
        text += str(predict(img))
        img = np.zeros((512,512,1), np.uint8)
        cv.putText(img, text, (10,500), cv.FONT_HERSHEY_PLAIN, 2, (155,155,155),2)
    elif k == ord('c'):
        text = ""
        img = np.zeros((512,512,1), np.uint8)
    elif k == 32:
        text += " "
    elif k == 27:
        break
    
    cv.imshow('image',img)
cv.destroyAllWindows()
