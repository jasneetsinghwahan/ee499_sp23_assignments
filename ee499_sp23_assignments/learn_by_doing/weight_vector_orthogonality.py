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
READDATAPATH = '../../ee499_ml_spring23/readData/'

# compute linear combination of input point
def model_1(x_p,w):
    # compute linear combination and return
    a = w[0] + np.dot(x_p.T,w[1:])
    return a.T

# a least squares function for linear regression
def least_squares(w,x,y):    
    cost = np.sum((model_1(x,w)-y)**2)
    return cost/float(y.size)

# using an automatic differentiator - like the one imported via the statement below - makes coding up gradient descent a breeze
from autograd import numpy as np
from autograd import value_and_grad 

# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g,alpha_choice,max_its,w, data, labels, inputs_flag = False):
    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
        
        # evaluate the gradient, store current weights and cost function value
        if (inputs_flag == True):
            cost_eval,grad_eval = gradient(w,data,labels)
        else:
            cost_eval,grad_eval = gradient(w)
            
        weight_history.append(w)
        cost_history.append(cost_eval)

        # take gradient descent step

# newtons method function - inputs: g (input function), max_its (maximum number of iterations), w (initialization)
def newtons_method(g,max_its,w,data, labels, inputs_flag = False):
    # compute gradient module using autograd
    gradient = grad(g)
    hess = hessian(g)
    
    # set numericxal stability parameter / regularization parameter
    epsilon = 10**(-7)
    #if 'epsilon' in kwargs:
    #    beta = kwargs['epsilon']

    # run the newtons method loop
    weight_history = [w]           # container for weight history
    
    if (inputs_flag == True):
        cost_history = [g(w,x,y)]
    else:
        cost_history = [g(w)] 
        
    for k in range(max_its):
        # evaluate the gradient and hessian
        if (inputs_flag == True):
            grad_eval = gradient(w,data,labels)
            hess_eval = hess(w, data, labels)
        else:
            grad_eval = gradient(w)
            hess_eval = hess(w)

        # reshape hessian to square matrix for numpy linalg functionality
        hess_eval.shape = (int((np.size(hess_eval))**(0.5)),int((np.size(hess_eval))**(0.5)))
      
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
        # return only the weight

def plot_contours(g, weight_history, title, data, labels, a, b, inputs_flag = False):
    weights_steps_x = np.array([i[0] for i in weight_history])
    weights_steps_y = np.array([i[1] for i in weight_history])
    x = y = np.arange(a, b, 0.1)
    X, Y = np.meshgrid(x, y)
    if (inputs_flag):
        zs = np.array([g(np.array([[x],[y]]), data, labels) for x,y in zip(np.ravel(X), np.ravel(Y))])
    else:
        zs = np.array([g(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    #ax.plot([0, 0], [-4.5,4.5], [0,0])
    fig = plt.figure(figsize=(5,5))
    cp = plt.contourf(X, Y, Z, cmap='coolwarm')
    plt.colorbar(cp)
    plt.xlabel(r'$w_1$', fontsize=12)
    plt.ylabel(r'$w_2$', fontsize=12)
    #plt.axhline(y=0, color='r', linestyle='--')
    #plt.axvline(x=0, color='r', linestyle='--')
    plt.plot(weights_steps_x, weights_steps_y, 'sk', markersize=4)
    plt.quiver(weights_steps_x[:-1], weights_steps_y[:-1], weights_steps_x[1:]-weights_steps_x[:-1], weights_steps_y[1:]-weights_steps_y[:-1], scale_units='xy', angles='xy', scale=1)
    plt.plot(weights_steps_x[-1], weights_steps_y[-1], 'sy')
    plt.title(title,fontsize=16)
    plt.show()

data = np.loadtxt(READDATAPATH + '2d_classification_data_v1_entropy.csv',delimiter = ',')
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

def model_classification_01(x_p,w):
    # compute linear combination and return
    a = w[0] + np.dot(x_p.T,w[1:])
    predictions = [sigmoid(l) for l in a]
    return predictions

g = sigmoid_least_squares; w = np.asarray([10.0,-5.0])[:,np.newaxis]; max_its = 100; alpha_choice = 1; #w = np.asarray([10.0,-5])[:,np.newaxis]
weight_history = normalized_gradient_descent(g, alpha = alpha_choice, max_its = max_its, w = w, data=x, labels=y, inputs_flag = True)

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