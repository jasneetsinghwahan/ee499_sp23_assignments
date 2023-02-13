# problem 2.9 Watt, et. al.:
# coordinate search with diminishing steplength

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

from user_functions import *

# kinda of gloabl variables as required by the question
target_value = 0
target_range = 1e-2

# objective_function
g = lambda w: 0.26*(w[0]**2 + w[1]**2) - 0.48*(w[0]*w[1])

# zero order coordinate search
def coordinate_search(g,alpha_choice,max_its,w):
    # construct set of all coordinate directions
    directions_plus = np.eye(np.size(w))    # creats a Identity matrix with no. of dimensions equal
    directions_minus = - np.eye(np.size(w)) # to no. of elements in w 
    directions = np.concatenate((directions_plus,directions_minus),axis=0)
        
    # run coordinate search
    weight_history = []         # container for weight history
    cost_history = []           # container for corresponding cost function history
    alpha = 0
    #for k in range(1,max_its+1): 
    # ignore max_its and continue till value - 0 = 1E-2
    k = 1
    while(1):       
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

            # continue running the results till the time you don't get closed to the target
            if((g(w_candidates[ind]) - target_value) < target_range):
                break
        k += 1
        
    # record weights and cost evaluation
    weight_history.append(w)
    cost_history.append(g(w))
    return k,weight_history,cost_history

# Main program
alpha_choice = 'diminishing'
max_its = 10
random_vector_history = []
steps_history = []
for i in range(1,5):
    np.random.seed(i) # set seed
    random_vector = np.random.randint(-10, 11, size=2)
    k, weight_history_2,cost_history_2 = coordinate_search(g,alpha_choice,max_its,random_vector)
    random_vector_history.append(random_vector)
    steps_history.append(k)
    plot_contours(g, weight_history_2, view=[20,50], flag3D = False, title = 'Coordinate Search')

# plotting plots
fig, (ax1) = plt.subplots(1, figsize=(10, 3))
magnitudes = [np.linalg.norm(vector) for vector in random_vector_history]
#ax1.plot(magnitudes, steps_history, 'k')
ax1.scatter(magnitudes, steps_history)
ax1.set_xlabel('start point', fontsize=12)
ax1.set_ylabel('no. of iterations', fontsize=10, labelpad = 0)
for i, txt in enumerate(zip(magnitudes, steps_history)):
    ax1.annotate(txt, (magnitudes[i], steps_history[i]))
plt.show()