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
        self.last = None
        self.bf = cv2.BFMatcher()
    def extract(self,img):
        #kps = []
        match =None
        sx = img.shape[1]/self.GX
        sy = img.shape[0]/self.GY
        """

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
                """
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray,3000,0.01,3)
        
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in corners]
        kps,dess =self.orb.compute(img,kps)
        if self.last is not None:
            match =self.bf.match(dess,self.last["des"])
        self.last = {"kps":kps,"des":dess}
        return [kps,dess,match]
                        

fe = FeatureExtractor()
def imgprocess(img):
    img = cv2.resize(img,(W,H))
    print img.shape
    
    
    kp,des,match = fe.extract(img)
    for p in kp:
        u,v = map(lambda x:int(round(x)),p.pt)
        cv2.circle(img,(u,v),color=(0,255,0),radius=3)
    
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
    
