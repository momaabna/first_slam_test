import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform
class FeatureExtractor(object):
    GX = 16/2
    GY = 12/2
    def __init__(self,K):
        self.orb = cv2.ORB_create(5000)
        self.last = None
        self.bf = cv2.BFMatcher()
        self.img =None
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.lavg=[]
    def extract(self,img):
        #kps = []
        match =None
        self.img= img
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
        ret =[]
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in corners]
        kps,dess =self.orb.compute(img,kps)
        pose =None
        if self.last is not None:
            match =self.bf.knnMatch(dess,self.last["des"],k=2)
            for m,n in match:
                if m.distance<0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last["kps"][m.trainIdx].pt
                    ret.append((kp1,kp2))
            if len(ret)>0:
                ret =np.array(ret)
                ret = self.normalize(ret)
                model, inliers = ransac((ret[:, 0],ret[:, 1]),
                        #FundamentalMatrixTransform,
                        EssentialMatrixTransform,
                        min_samples=8,
                        residual_threshold=0.001,
                        max_trials=100)
                
                ret = ret[inliers]
                
                pose =self.extractRt(model.params)
                

                
                
                
                #print f_l,np.mean(self.lavg)
            #match = zip([kps[m.QueryIdx] for m in match],[self.last["kps"][m.trainIdx] for m in match])
        self.last = {"kps":kps,"des":dess}
        return ret,pose
    def extractRt(self,E):
        R=None
        w = np.mat([[0,1,0],[-1,0,0],[0,0,1]])
        u,s,vt =np.linalg.svd(E)
        if np.linalg.det(vt)<0:
            vt *=-1
        R = u * w*vt;
        
        if np.linalg.det(R)<0:
            R = u * w.T *vt
        t = u[:,2]
        
        pose = np.concatenate([R,t.reshape(3,1)],axis=1)
        pose = np.concatenate([pose,np.array([[0],[0],[0],[1]]).T],axis=0)
        return pose
    def add_ones(self,x):
        return np.concatenate([x,np.ones((x.shape[0],1))],axis=1)
    def denormalize(self,pt):
        re = np.dot(self.K,np.array([pt[0],pt[1],1]).T)
        re /=re[2]
        return int(round(re[0])),int(round(re[1]))
        #return int(round(pt[0]+self.img.shape[0]/2)),int(round(pt[1]+self.img.shape[1]/2))
    def normalize(self,ret):
        ret[:,0,:] =np.dot(self.Kinv,self.add_ones(ret[:,0,:]).T).T[:,0:2]#self.img.shape[0]/2
        ret[:,1,:] =np.dot(self.Kinv,self.add_ones(ret[:,1,:]).T).T[:,0:2]#ret[:,:,1]#self.img.shape[1]/2
        return ret
