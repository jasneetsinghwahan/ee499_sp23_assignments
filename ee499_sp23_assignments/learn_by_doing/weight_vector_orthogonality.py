"""
This module is testing that weight vector is orthogonal to the seperatability boundary
ALWAYS REMEMBER TO CHECK THE CURRENT PATH
"""
#import libraries
import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from autograd import grad 
from autograd import hessian
import math
import copy
import pandas as pd

# degfault path is different from the current path
import os


# Specify the path to go to from default directory
CURRENT_DIR = os.getcwd()
SUB_DIR = "ee499_sp23_assignments/learn_by_doing"
NEW_DIR = os.path.join(CURRENT_DIR, SUB_DIR)
os.chdir(NEW_DIR)
# load in dataset
readDataPath = '../../ee499_ml_spring23/readData/'

def normalized_gradient_descent(g,alpha,max_its,w, data, labels, inputs_flag = True):
    # compute the gradient of our input function - note this is a function too!
    gradient = grad(g)
    weight_history = [w] 
    # run the gradient descent loop
    best_w = w        # weight we return, should be the one providing lowest evaluation
    if (inputs_flag):
        best_eval = g(w,data,labels)       # lowest evaluation yet
    else:
        best_eval = g(w)
    
    for k in range(max_its):
        # evaluate the gradient, compute its length
        if (inputs_flag):
            grad_eval = gradient(w,data,labels)
        else:
            grad_eval = gradient(w)
        grad_norm = np.linalg.norm(grad_eval)
        
        # check that magnitude of gradient is not too small, if yes pick a random direction to move
        if grad_norm == 0:
            # pick random direction and normalize to have unit legnth
            grad_eval = 10**-6*np.sign(2*np.random.rand(len(w)) - 1)
            grad_norm = np.linalg.norm(grad_eval)
            grad_eval /= grad_norm
    
        # take gradient descent step
        w = w - alpha*grad_eval
        
        weight_history.append(w) 
        # return only the weight providing the lowest evaluation
        if (inputs_flag):
            test_eval = g(w,data,labels)
        else:
            test_eval = g(w)
            
        if test_eval < best_eval:
            best_eval = test_eval
            best_w = w

    #return best_w.tolist()
    return weight_history

data = np.loadtxt(readDataPath + '2d_classification_data_v1_entropy.csv',delimiter = ',')
x = data[:-1,:] # up to last row (exclusive) of data & all columns
y = data[-1:,:] # last row of data & all columns
assert x.shape == y.shape, f'Shape mismatch between X and Y'

def sigmoid(t):
    return 1/(1 + np.exp(-t))

# sigmoid non-convex logistic least squares cost function
def sigmoid_least_squares(w,x,y):
    a = w[0] + np.dot(x.T,w[1:])
    predictions = np.array([sigmoid(l) for l in a]).T
    cost = sum((predictions.flatten()-y.flatten())**2)
    return cost/y.size

x = data[:-1,:] # up to last row (exclusive) of data & all columns
y = data[-1:,:] # last row of data & all columns

#w_hist = gradient_descent(g = sigmoid_least_squares,w = w,version = 'normalized',max_its = 900, alpha = 1)
g = sigmoid_least_squares; w = np.asarray([10.0,-5.0])[:,np.newaxis]; max_its = 1000; alpha_choice = 1; #w = np.asarray([10.0,-5])[:,np.newaxis]
#weight_history,cost_history = gradient_descent(g,alpha_choice,max_its,w, data = x, labels = y, inputs_flag=True)
weight_history = normalized_gradient_descent(g, alpha = alpha_choice, max_its = max_its, w = w, data=x, labels=y, inputs_flag = True)

def model_classification_01(x_p,w):
    # compute linear combination and return
    a = w[0] + np.dot(x_p.T,w[1:])
    predictions = [sigmoid(l) for l in a]
    return predictions

plt.figure(figsize=(10, 5))
plt.scatter(x,y,s=100,facecolors='k', edgecolors='k')
plt.xlabel('X', fontsize = 12)
plt.ylabel('Y', fontsize = 12)
plt.title("Classification Problem with Sigmoid Least Square - Gradient Descent", fontsize=16)
x_line = np.linspace(-2,7,100).reshape((1,100))
y_line = model_classification_01(x_line,weight_history[-1])
plt.plot(x_line[0],y_line,'r-',linewidth=3)
plt.xlim(left=-2, right=6.5)
plt.show()