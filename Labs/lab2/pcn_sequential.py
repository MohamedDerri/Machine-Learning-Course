# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

from numpy import *

class pcn:
	""" A basic Perceptron (the same pcn.py except with the weights printed
	and it does not reorder the inputs)"""
	
	def __init__(self,inputs,targets):
		""" Constructor """
		# Set up network size
		if ndim(inputs)>1:
			self.nIn = shape(inputs)[1]
		else: 
			self.nIn = 1
	
		if ndim(targets)>1:
			self.nOut = shape(targets)[1]
		else:
			self.nOut = 1

		self.nData = shape(inputs)[0]
	
		# Initialise network
		self.weights = random.rand(self.nIn+1,self.nOut)*0.1-0.05

	def pcntrain(self,inputs,targets,eta,nIterations):
		""" Train the thing """	
		# Add the inputs that match the bias node
		inputs = concatenate((inputs,-ones((self.nData,1))),axis=1)
	
		# Training
		for n in range(nIterations):
			for i in range(self.nData):
				# Compute activation
				activation = dot(inputs[i], self.weights)
				# Apply step function
				output = 1 if activation >= 0 else 0
				# Compute error
				error = targets[i] - output
				# Update weights
				self.weights += eta * error * inputs[i].reshape(-1,1)
				print("final output : ", output)


