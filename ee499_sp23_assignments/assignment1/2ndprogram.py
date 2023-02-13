# Assignment 1: Problem 2
# Zero-order Coordinate search vs coordinate descent
# Potentially coordinate descent should do more updates to 'w' as compared to coordinate search
# because coordinate descent can update with every evaluation of a direction (in N direction/dimensional)
# space, where coordinate search only potentially updates once per step k
# thus, potentially coordinate descent can update k times N directions where updates in coordinate search
# could k steps only     

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from matplotlib import cm
import copy
# import automatic differentiator to compute gradient module
from autograd import grad
from PIL import Image
from scipy.signal import find_peaks

###################################################################
# contour plots function
###################################################################
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
    fig = plt.figure(figsize=(5,5))
    cp = plt.contourf(X, Y, Z, cmap='coolwarm')
    plt.colorbar(cp)
    plt.xlabel(r'$w_1$', fontsize=12)
    plt.ylabel(r'$w_2$', fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.plot(weights_steps_x, weights_steps_y, 'sk', markersize=4)
    plt.quiver(weights_steps_x[:-1], weights_steps_y[:-1], weights_steps_x[1:]-weights_steps_x[:-1], weights_steps_y[1:]-weights_steps_y[:-1], scale_units='xy', angles='xy', scale=1)
    plt.plot(weights_steps_x[-1], weights_steps_y[-1], 'sy')
    plt.title(title,fontsize=16)
    plt.show()

###################################################################
# zero order coordinate search
###################################################################
def coordinate_search(g,alpha_choice,max_its,w):
    # construct set of all coordinate directions
    #directions_plus = np.eye(np.size(w),np.size(w))
    directions_plus = np.eye(np.size(w))
    #directions_minus = - np.eye(np.size(w),np.size(w))
    directions_minus = - np.eye(np.size(w))
    directions = np.concatenate((directions_plus,directions_minus),axis=0)
        
    # run coordinate search
    weight_history = []         # container for weight history
    cost_history = []           # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):        
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
            
        # record weights and cost evaluation
        weight_history.append(w)
        cost_history.append(g(w))
        
        ### pick best descent direction
        # compute all new candidate points
        w_candidates = w + alpha*directions
        
        # evaluate all candidates
        evals = np.array([g(w_val) for w_val in w_candidates])

        # if we find a real descent direction take the step in its direction
        ind = np.argmin(evals)
        if g(w_candidates[ind]) < g(w):
            # pluck out best descent direction
            d = directions[ind,:]
        
            # take step
            w = w + alpha*d
        
    # record weights and cost evaluation
    weight_history.append(w)
    cost_history.append(g(w))
    return weight_history,cost_history

###################################################################
# zero order coordinate descent
###################################################################
def coordinate_descent_zero_order(g,alpha_choice,max_its,w):  
    # run coordinate search
    N = np.size(w)
    weight_history = []         # container for weight history
    cost_history = []           # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):        
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
        
        # random shuffle of coordinates
        c = np.random.permutation(N)
        
        # forming the direction matrix out of the loop
        cost = g(w)
        
        # loop over each coordinate direction
        for n in range(N):
            direction = np.zeros((N,1)).flatten()
            direction[c[n]] = 1
    
            # record weights and cost evaluation
            weight_history.append(w)
            cost_history.append(cost)

            # evaluate all candidates
            evals =  [g(w + alpha*direction)]
            evals.append(g(w - alpha*direction))
            evals = np.array(evals)

            # if we find a real descent direction take the step in its direction
            ind = np.argmin(evals)
            if evals[ind] < cost_history[-1]:
                # take step
                w = w + ((-1)**(ind))*alpha*direction
                cost = evals[ind]
        
    # record weights and cost evaluation
    weight_history.append(w)
    cost_history.append(g(w))
    return weight_history,cost_history


###################################################################
# g(w) function
###################################################################
def g(w):
    return 0.26*(w[0]**2 + w[1]**2) - 0.48*w[0]*w[1]

###################################################################
# Main program
###################################################################
alpha_choice = 'diminishing'; w = np.array([3,4]);

###################################################################
# running coordinate search algo. multiple iterations
###################################################################
# calculating no. of times to run which is <No._of_dimensions> times <increasing_power_of_2> 
pow_of_2 = [2**i for i in range(1,10)]
pow_of_2_arr = np.array(pow_of_2)
max_its = len(w) * pow_of_2_arr;

# run coordinate search algorithm multiple times
no_of_upd_1 = np.zeros(len(max_its))

for case, itr_times in enumerate(max_its):
    # run coordinate search algorithm
    weight_history_1,cost_history_1 = coordinate_search(g,alpha_choice,itr_times,w)
    no_of_upd_1[case] = len(weight_history_1)
print(f'no of updates {no_of_upd_1}')
#plot_contours(g, weight_history_1, view=[20,50], flag3D = False, title = 'Coordinate Search')

###################################################################
# run coordinate descent algorithm multiple times
###################################################################
w = np.array([3,4]); 

alpha_choice = 'diminishing'; 

# run coordinate descent zero order multiple times
no_of_upd_2 = np.zeros(len(max_its))

for case, itr_times in enumerate(max_its):
    # run coordinate descent zero order algorithm
    weight_history_2,cost_history_2 = coordinate_descent_zero_order(g,alpha_choice,itr_times,w)
    no_of_upd_2[case] = len(weight_history_2)
print(f'no of updates {no_of_upd_2}')

ratio_upd = np.divide(no_of_upd_2,no_of_upd_1)

# plotting plots
fig, (ax1) = plt.subplots(1, figsize=(10, 3))
ratio_upd_rounded = [round(val, 2) for val in ratio_upd]
ax1.plot(max_its, ratio_upd_rounded, 'k')
ax1.scatter(max_its, ratio_upd_rounded)
ax1.set_xlabel('no. of iterations', fontsize=12)
ax1.set_ylabel('ratio of updates in coordiante descent to coordinate search', fontsize=10, labelpad = 0)
y_label = plt.gca().yaxis.get_label()
for i, txt in enumerate(zip(max_its, ratio_upd_rounded)):
    ax1.annotate(txt, (max_its[i], ratio_upd_rounded[i]))
plt.show()

#plot_contours(g, weight_history_2, view=[20,50], flag3D = False, title = 'Coordinate Descent')