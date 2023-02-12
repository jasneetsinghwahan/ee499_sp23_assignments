# Problem 3.1 sub-part 4 of Watt, et. al.:
# this program only generates the contour plot of the function
# and not for the derivative
  
import numpy as np
import matplotlib.pyplot as plt
import math         # Use math.log function from the 'math' module
import time
from matplotlib import cm
import copy
# import automatic differentiator to compute gradient module
from autograd import grad
from PIL import Image
from scipy.signal import find_peaks
#from scipy.special import sech     # to calculate the hyperbolic secant (sech) function
#from scipy.special import cosh      # to calculate the hyperbolic secant (sech) function
from math import cosh

def plot_contours(g, weight_history, view, flag3D, title):
    weights_steps_x = np.array([i[0] for i in weight_history])
    weights_steps_y = np.array([i[1] for i in weight_history])
    x = y = np.arange(-4.5, 4.5, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([g(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    if (flag3D):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111,projection='3d')
        ax.quiver(weights_steps_x[:-1], weights_steps_y[:-1], np.zeros(weights_steps_x[:-1].shape[0]), weights_steps_x[1:]-weights_steps_x[:-1], weights_steps_y[1:]-weights_steps_y[:-1], np.zeros(weights_steps_x[:-1].shape[0]),\
            color='k')
        ax.grid(False)
        ax.plot_surface(X, Y, Z, alpha=0.8, cmap=cm.coolwarm, linewidth=10, antialiased=False)
        ax.view_init(view[0], view[1])
        ax.set_xlabel(r'$w_1$')
        ax.set_ylabel(r'$w_2$')
        ax.set_zlabel(r'$g(w)$')
        ax.set_title(title,fontsize=16)
    #ax.plot([0, 0], [-4.5,4.5], [0,0])
    #fig = plt.figure(figsize=(5,5))
    #cp = plt.contourf(X, Y, Z, cmap='coolwarm')
    #plt.colorbar(cp)
    #plt.xlabel(r'$w_1$', fontsize=12)
    #plt.ylabel(r'$w_2$', fontsize=12)
    #plt.axhline(y=0, color='r', linestyle='--')
    #plt.axvline(x=0, color='r', linestyle='--')
    #plt.plot(weights_steps_x, weights_steps_y, 'sk', markersize=4)
    #plt.quiver(weights_steps_x[:-1], weights_steps_y[:-1], weights_steps_x[1:]-weights_steps_x[:-1], weights_steps_y[1:]-weights_steps_y[:-1], scale_units='xy', angles='xy', scale=1)
    #plt.plot(weights_steps_x[-1], weights_steps_y[-1], 'sy')
    #plt.title(title,fontsize=16)
#fig, (ax1) = plt.subplots(3, 1, figsize=(10, 9))
#fig.subplots_adjust(wspace=0.5, hspace=0.5)

# the function g sub-part 4
# g(w)
# input: vector quantity in N dimensions
# output: scaler
def objective_vector(w):
	return 0.5*np.dot(w.T,np.dot(C, w)) + np.dot(b.T,w)

# the derivative of g sub-part 1
def gradientg_objvector(w):
    return np.dot(C,w)+b

C = np.array([[2,1],[1,3]])
b = np.array([[1],[1]])
#fig, (ax1) = plt.subplots(1, 1, figsize=(10, 9))
#fig.subplots_adjust(wspace=0.5, hspace=0.5)
#ax1.set_title('g(w) = w.log(w) + (1-w).log(w)', color = 'k',loc='left')
#ax1.annotate('d(g(w))/dw = log(w/(1-w))', xy=(1,1), \
#    xycoords="axes fraction", xytext=(10,10), textcoords="offset points", \
#        ha="right", va="top",color = 'g')
#ax1.annotate('d(g(w))/dw = log(w/(1-w))', xy=(1,0.5),ha='right',va='top',color = 'g')
##r_min, r_max = -10.0, 10.0

import numpy as np

# Define grid dimensions
row_count = 2
column_count = 20
start = -10
stop = 10
step = (stop - start) / (column_count - 1)

# Generate grid using np.linspace()
inputs = np.linspace(start, stop, column_count)
inputs = np.vstack([inputs]*row_count)
function = []

#inputs = np.random.unform(low=r_min, high=r_max, size=(20,20))
for i in range(20):
    function.append(objective_vector(inputs[:,i]))

#function = [objective_vector(x,C,b) for x in inputs]
#ax1.plot(inputs, function, 'k')
#ax1.set_xlabel('w', fontsize=12)
#ax1.set_ylabel('g(w)', fontsize=12)
#derivative = [gradientg_objvector(x) for x in inputs]
#ax1.plot(inputs, derivative, 'g')
#ax1.plot(0.5,0,'gD' )
plot_contours(objective_vector,inputs,view=[20,100],flag3D=True, title='matrix function')
#plot_contours(gradientg_objvector,inputs,view=[20,100],flag3D=True, title='derivative function')
plt.show()