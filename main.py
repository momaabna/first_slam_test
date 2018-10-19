import cv2
import numpy as np
import pygame
W = 1920/2
H = 1080/2
cv2.startWindowThread()
cv2.namedWindow("image")
# Initiate SIFT detector


class FeatureExtractor(object):
    GX = 16/2
    GY = 12/2
    def __init__(self):
        self.orb = cv2.ORB(5000)

    def extract(self,img):
        kps = []
        
        sx = img.shape[1]/self.GX
        sy = img.shape[0]/self.GY
        for ry in range(0,img.shape[0],sy):
            for rx in range(0,img.shape[1],sx):
                img_chunk =  img[ry:ry+sy,rx:rx+sx]
                #print img_chunk.shape
                kp = self.orb.detect(img_chunk,None)
                #print kp
                if len(kp)>0:
                    for p in kp:
                        p.pt=(p.pt[0]+rx,p.pt[1]+ry)
                        kps.append(p)
                dess =self.orb.compute(img,kps)
        return [kps,dess]
                        

fe = FeatureExtractor()
def imgprocess(img):
    img = cv2.resize(img,(W,H))
    print img.shape
    
    
    kp,des = fe.extract(img)
    for p in kp:
        u,v = map(lambda x:int(round(x)),p.pt)
        cv2.circle(img,(u,v),color=(0,255,0),radius=5)
    
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
    
