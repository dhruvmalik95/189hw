
import random
import time


import numpy as np
import numpy.linalg as LA


from numpy.linalg import inv
from numpy.linalg import det

from projection import Project2D, Projections


class QDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.01
		self.NUM_CLASSES = len(class_labels)

	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

		'''
		self.mapp = {}
		for i in range(0, len(X)):
			x = X[i]
			y = Y[i]
			
			if y in self.mapp:
				self.mapp[y].append(x)
			else:
				self.mapp[y] = [x]

		self.mapp_parameters = {}
		
		for y in self.mapp:
			l = self.mapp[y]
			a,b=np.shape(np.array(l))
			mean = np.mean(l, axis=0)

			# cov = np.zeros((b,b))
			# for j in l:
			# 	cov = cov + ((j-mean).T).dot((j-mean))
			# cov = cov+self.reg_cov*np.identity(len(cov))
			# cov = cov/len(l)

			cov = ((l-mean).T).dot(l-mean)
			cov = cov + self.reg_cov*np.identity(b)
			cov = cov/len(l)
			
			tup = (mean, cov)
			self.mapp_parameters[y] = tup

	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		best_value = float("-inf")
		best_class = "hi"
		
		for y in self.mapp_parameters:
			mean, cov = self.mapp_parameters[y]
			v = -1*np.dot((x-mean).T, np.dot(np.linalg.inv(cov), (x-mean))) - np.log(det(cov))
			if v >= best_value:
				best_class = y
				best_value = v
		return best_class





	