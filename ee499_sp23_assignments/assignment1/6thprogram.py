# problem 2.10 Watt, et. al.:
# curse of dimensionality revisited for coordinate search and coordinate descent algo.

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
import itertools
from user_functions import *

def g(w):
    return np.dot(w, w.T) + 2

###################################################################
# zero order coordinate search
# but instead of traversing through various iterations, it just counts the number of directions that 
# had their evaluations less than g(w)
###################################################################
def coordinate_search_custom(g,alpha_choice,max_its,w):
    #tmp_variable = g(w)
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
        count = 0
        if g(w_candidates[ind]) < g(w):
            # pluck out best descent direction
            d = directions[ind,:]
        
            # take step
            w = w + alpha*d

        #for i in range(len(evals)):
        #    if evals[i] < tmp_variable:
        #        # pluck out best descent direction
        #        count += 1        
        
        # record weights and cost evaluation
    weight_history.append(w)
    cost_history.append(g(w))
    return weight_history,cost_history

###################################################################
# zero order coordinate descent
###################################################################
def coordinate_descent_zero_order_custom(g,alpha_choice,max_its,w):  
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
        count = 0
        
        # loop over each coordinate direction
        for n in range(N):
            direction = np.zeros((N,1)).flatten()
            direction[c[n]] = 1

            # record weights and cost evaluation from first time as well as previous iterations
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

            # if we find a real descent direction take the step in its direction
            count += len([i for i in evals if i < cost]) 
        
    # record weights and cost evaluation
    weight_history.append(w)
    cost_history.append(g(w))
    return weight_history,cost_history
    #return count

###################################################################
# main function
###################################################################
# coordinate search
###################################################################
#n_dim_lst = [i for i in range(1,10)]
n_dim_lst = [2]
alpha_choice = 'diminishing'
P_points_lst = [100]
max_its = 20
fig, axs = plt.subplots(2,len(P_points_lst),figsize = (15,6))
#fig.suptitle('coordiante search')
plt.subplots_adjust(hspace=0.5)
#plt.figtext(0.08, 0.5, "count of evaluations less than objective function", ha="center", va="center", rotation="vertical")
for itr, num_points in enumerate(P_points_lst):
    count_history = []
    for cnt_dim, dim_val in enumerate(n_dim_lst):
        # generate a random_vector with dim_val dimensions and range between 2 and -2
        random_vector = 2 * np.random.randn(dim_val) - 1
        #random_vector = np.random.randint(-1, 1, size=dim_val)
        # run coordinate search algorithm
        weight_history_1,cost_history_1 = coordinate_search_custom(g,alpha_choice,max_its,random_vector)
        #count_history.append(count)
    # plotting the learning curve
    k_steps = list(range(1,len(cost_history_1)+1))
    cost_history_1_rounded = [round(val, 2) for val in cost_history_1]
    axs[0].plot(k_steps,cost_history_1_rounded, 'g')
    axs[0].scatter(k_steps,cost_history_1_rounded)
    ##axs[0].set_yscale('log')
    axs[0].set_xlabel('k Steps', fontsize=8)
    axs[0].set_ylabel('cost function evaluation', fontsize=10)
    axs[0].set_title(f'learning curve for coordinate search')
    for i, txt in enumerate(zip(k_steps, cost_history_1_rounded)):
        axs[0].annotate(txt, (k_steps[i], cost_history_1_rounded[i])) 
    plot_contours(g, weight_history_1, view=[20,50], flag3D = False, title = 'Coordinate Search')


###################################################################
# zero-order coordinate descent
###################################################################
#fig.suptitle('coordiante descent')
for itr, num_points in enumerate(P_points_lst):
    count_history = []
    for cnt_dim, dim_val in enumerate(n_dim_lst):
        # generate a random_vector with dim_val dimensions and range between 2 and -2
        random_vector = 2 * np.random.randn(dim_val) - 1
        #random_vector = np.random.randint(-10, 11, size=dim_val)
        # run coordinate search algorithm
        weight_history_2,cost_history_2 = coordinate_descent_zero_order_custom(g,alpha_choice,num_points,random_vector)
        #count_history.append(count)
    # plotting the learning curve
    k_steps = list(range(1,len(cost_history_2)+1))
    cost_history_2_rounded = [round(val, 2) for val in cost_history_2]
    axs[1].plot(k_steps,cost_history_2_rounded, 'g')
    axs[1].scatter(k_steps,cost_history_2_rounded)
    ##axs[1].set_yscale('log')
    axs[1].set_xlabel('k Steps', fontsize=8)
    axs[1].set_ylabel('cost function evaluation', fontsize=10)
    axs[1].set_title(f'learning curve for coordinate descent')
    #for i, txt in enumerate(zip(k_steps, cost_history_2_rounded)):
    #    axs[1].annotate(txt, (k_steps[i], cost_history_2_rounded[i])) 
    plot_contours(g, weight_history_2, view=[20,50], flag3D = False, title = 'Coordinate Descent')


###################################################################
# zero-order coordinate descent - plotting min. g(w) vs no. of iterations aka computational budget
###################################################################
#fig.suptitle('coordiante descent')
alpha_choice = 'diminishing'
fig, bxs = plt.subplots(2,1,figsize = (15,6))
fig.suptitle('min. g(w) vs no. of iterations aka computational budget')
plt.subplots_adjust(hspace=0.5)
P_points_lst = [10, 20, 40]
master_cost_history_2 = []
random_vector = 10 * np.random.randn(n_dim_lst[0]) - 10
for itr, num_points in enumerate(P_points_lst):
    count_history = []
    for cnt_dim, dim_val in enumerate(n_dim_lst):
        # generate a random_vector with dim_val dimensions and range between 2 and -2
        #random_vector = np.random.randint(-10, 11, size=dim_val)
        # run coordinate search algorithm
        weight_history_2,cost_history_2x = coordinate_descent_zero_order_custom(g,alpha_choice,num_points,random_vector)
        #count_history.append(count)
    master_cost_history_2.append(cost_history_2x[-1])

# plotting min. g(w) vs no. of iterations aka computational budget 
master_cost_history_2_rounded = [round(val, 2) for val in master_cost_history_2]
bxs[1].plot(P_points_lst,master_cost_history_2_rounded, 'g')
bxs[1].scatter(P_points_lst,master_cost_history_2_rounded)
##bxs[1].set_yscale('log')
bxs[1].set_xlabel('Count of evaluations', fontsize=8)
bxs[1].set_ylabel('cost function evaluation', fontsize=10)
bxs[1].set_title(f'min. g(w) vs. computational budget for coordinate descent')
for i, txt in enumerate(zip(P_points_lst, master_cost_history_2_rounded)):
        axs[1].annotate(txt, (P_points_lst[i], master_cost_history_2_rounded[i])) 


###################################################################
# coordinate search- plotting min. g(w) vs no. of iterations aka computational budget
###################################################################
#n_dim_lst = [i for i in range(1,10)]
#plt.figtext(0.08, 0.5, "count of evaluations less than objective function", ha="center", va="center", rotation="vertical")
master_cost_history_1 = []
for itr, num_points in enumerate(P_points_lst):
    count_history = []
    for cnt_dim, dim_val in enumerate(n_dim_lst):
        # generate a random_vector with dim_val dimensions and range between 2 and -2
        #random_vector = 2 * np.random.randn(dim_val) - 1
        #random_vector = np.random.randint(-1, 1, size=dim_val)
        # run coordinate search algorithm
        weight_history_1x,cost_history_1x = coordinate_search_custom(g,alpha_choice,num_points,random_vector)
        #count_history.append(count)
    master_cost_history_1.append(cost_history_1x[-1])

# plotting min. g(w) vs no. of iterations aka computational budget 
master_cost_history_1_rounded = [round(val, 2) for val in master_cost_history_1]
bxs[0].plot(P_points_lst,master_cost_history_1_rounded, 'g')
bxs[0].scatter(P_points_lst,master_cost_history_1_rounded)
##bxs[0].set_yscale('log')
bxs[0].set_xlabel('Count of evaluations', fontsize=8)
bxs[0].set_ylabel('cost function evaluation', fontsize=10)
bxs[0].set_title(f'min. g(w) vs. computational budget for coordinate search')
for i, txt in enumerate(zip(P_points_lst, master_cost_history_1_rounded)):
        axs[1].annotate(txt, (P_points_lst[i], master_cost_history_1_rounded[i])) 

# plotting 
plt.show()