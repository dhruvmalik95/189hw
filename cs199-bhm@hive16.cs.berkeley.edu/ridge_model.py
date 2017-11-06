
from numpy.random import uniform
import random
import time


import numpy as np
import numpy.linalg as LA

import sys

from sklearn.linear_model import Ridge

from utils import create_one_hot_label


class Ridge_Model(): 

	def __init__(self,class_labels):

		###RIDGE HYPERPARAMETER
		self.lmda = 1.0
		self.NUM_CLASSES = len(class_labels)
		



	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

		'''
		Y = create_one_hot_label(Y, self.NUM_CLASSES)
		self.solver = Ridge(self.lmda)
		self.solver.fit(X, Y)
		
		
		

	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		sol = self.solver.predict(np.matrix(x))
		return np.argmax(sol)



	