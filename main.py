import sys
sys.path.append("build/Pangolin/build/src")
import cv2
import numpy as np
import pygame
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from display import Display3D
#from frames import frame
from FeatureExtractor import FeatureExtractor

W = 1920//2
H = 1080//2
#cv2.startWindowThread()
#cv2.namedWindow("image")
F=750
K = np.array([[F, 0, W/2],[0,F,H/2],[0,0,1]])
fe = FeatureExtractor(K)
p3d = [];
poses = []
init_pose =np.eye(4)
poses.append(init_pose)
fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax1 = fig.add_subplot(212)
disp3d = Display3D()
def imgprocess(img):
    img = cv2.resize(img,(W,H))
    #print img.shape
    match,pose = fe.extract(img)
    #print( pose)
    if match is None or pose is None:
        return
    #print pose
    newpose = np.dot(poses[-1],pose)
    pp3d = ex3d(match,poses[-1],newpose)
    p3d.append(pp3d)
    print( len(p3d))
    #ax.scatter(pp3d[:,0], pp3d[:,1], pp3d[:,2], c='r', marker='o')
    #plt.show()
    disp3d.paint(pp3d,newpose)
    poses.append(newpose)
    print( len(match))
    for p1,p2 in match:
        u1,v1 = fe.denormalize(p1)
        u2,v2 = fe.denormalize(p2)
        cv2.line(img,(u1,v1),(u2,v2),color=(0,255,0))
    #for m in  match:
        #print match
    #ax1.imshow(img)
    cv2.imshow("image",img)
    cv2.waitKey(1)

def ex3d(match,pose1,pose2):
    pts1 = match[:,0,:]
    pts2 = match[:,1,:]
    ret = np.zeros((pts1.shape[0], 4))
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    #good_pts4d &= np.abs(pts4d[:, 3]) != 0
    ret /= ret[:, 3:]
    return ret
    #A = np.concatenate([pose1,pose2],axis=0)
    #A = np.zeros((4,4))
    #A[0] = p[0][0] * pose1[2] - pose1[0]
    #A[1] = p[0][1] * pose1[2] - pose1[1]
    #A[2] = p[1][0] * pose2[2] - pose2[0]
    #A[3] = p[1][1] * pose2[2] - pose2[1]
    #_, _, vt = np.linalg.svd(A)
    #ret[i] = vt[3]
    #p = fe.normalize(match)
    #for (x1,y1),(x2,y2) in p :
    #    B = np.array([[x1],[y1],[1],[x2],[y2],[1]])
    #    X = np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,B))
    #return X






if __name__ =="__main__":
    cap =cv2.VideoCapture("./data/test_countryroad.mp4")
    while cap.isOpened():
        re,img = cap.read()
        imgprocess(img)
    else:
        pass
    cap.release()
    cv2.destroyAllWindows()

