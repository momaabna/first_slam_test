import numpy as np

class frame(object):
	def __init__(self,K,img,kps,des,pose):
		self.pose =pose
		self.img =img
		self.K =K
		self.Kinv = np.linalg.inv(K)
		self.kps =kps
		self.des =des


class frames(object):
	def __init__(self)
		self.frames =[]
		pass
	def add_frame(frame):
		self.frames.append(frame)
	def get_frame(id):
		return frames[id]
 
