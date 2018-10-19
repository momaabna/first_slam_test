import cv2
import numpy as np
import pygame
W = 1920/2
H = 1080/2
cv2.startWindowThread()
cv2.namedWindow("image")
def imgprocess(img):
    img = cv2.resize(img,(W,H))
    print img.shape
    return img
    
    









if __name__ =="__main__":
    cap =cv2.VideoCapture("test_countryroad.mp4")
    while cap.isOpened():
        re,img = cap.read()
        image =imgprocess(img)
        cv2.imshow("image",image)
        cv2.waitKey(1)
        #key = cv2.waitKey(0) & 0xFF
        #if key == ord("q"):
        #    break
    else:
        pass
    cap.release()
    cv2.destroyAllWindows()
    
