import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.svm import LinearSVC
from projection import Project2D, Projections


class LDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.001
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
		
		big_l = []
		for y in self.mapp:
			l = self.mapp[y]
			a,b=np.shape(np.array(l))
			mean = np.mean(l, axis=0)

			big_l.append(l - mean)

			self.mapp_parameters[y] = mean	

		big_l = np.vstack(big_l)
		cov = (big_l.T).dot(big_l)
		cov = cov/len(X)
		cov = cov + self.reg_cov*np.identity(len(cov))
		self.avg_cov = cov

	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		best_value = float("-inf")
		best_class = "hi"
		
		for y in self.mapp_parameters:
			mean = self.mapp_parameters[y]
			v = -1*np.dot((x-mean).T, np.dot(np.linalg.inv(self.avg_cov), (x-mean)))
			if v >= best_value:
				best_class = y
				best_value = v
		return best_class


	