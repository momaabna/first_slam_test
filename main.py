import cv2
import numpy as np
import pygame
from FeatureExtractor import FeatureExtractor
W = 1920/2
H = 1080/2
cv2.startWindowThread()
cv2.namedWindow("image")
F=750
K = np.array([[F, 0, W/2],[0,F,H/2],[0,0,1]])
fe = FeatureExtractor(K)
def imgprocess(img):
    img = cv2.resize(img,(W,H))
    #print img.shape
    match,pose = fe.extract(img)
    print( pose)
    if match is None:
        return
    print( len(match))
    for p1,p2 in match:
        u1,v1 = fe.denormalize(p1)
        u2,v2 = fe.denormalize(p2)
        cv2.line(img,(u1,v1),(u2,v2),color=(0,255,0))
    #for m in  match:
        #print match
    cv2.imshow("image",img)
    cv2.waitKey(1)








if __name__ =="__main__":
    cap =cv2.VideoCapture("./data/test_countryroad.mp4")
    while cap.isOpened():
        re,img = cap.read()
        imgprocess(img)
    else:
        pass
    cap.release()
    cv2.destroyAllWindows()

