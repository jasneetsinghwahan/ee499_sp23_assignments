# Problem 2.1 Watt, et. al.:
# for creating linear space, i took the help of ChatGPT3
# exact query in chatgpt3 was "construct a linear space with varying number of dimensions"
# for uniform sampling, since size_domain was 1000 and grid would be extremely large\
# i have restricted the number of dimensions in random samlimg to 1000 max and in\
# random sampling to 30. Furtehrmore, no. of dimensions in random_sampling are 1 to 5

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

# g(w)
# input: vector quantity in N dimensions
# output: scaler
# defining the function g(w) that accepts vectors and does 
# operation on vectors 
def objective_vector(w):
	return np.dot(w, w.T)

# multi-dimensional random sampling
def multiDimensional_sampling(number_samples, n_dimension, case, size_domain):
    r_min = -1; r_max = 1;
    X = np.linspace(r_min, r_max, size_domain)
    accepted_samples = []
    accepted_samples_objective = []
    minimum = 1e30
    k = 0       # Count no. of iterations
    stuck_flag = 0
    stuck_iteration = 1e6
    while k < number_samples:
        stuck_flag = stuck_flag + 1
        if stuck_flag == stuck_iteration:
            print(f"Case {case+1} exited since algorithm was stuck after {stuck_iteration} iterations")
            return accepted_samples, accepted_samples_objective
        sample = np.zeros(n_dimension)
        for i in range(0,n_dimension):
            sample[i] = np.random.choice(X)
        if objective_vector(sample) < minimum:
            stuck_flag = 0
            accepted_samples.append(sample)
            minimum = objective_vector(sample)
            accepted_samples_objective.append(minimum)
            if abs(minimum - objective_value) <= 1e-2:
                print(f"Case {case+1} convergence is reached with {k+1} samples")
                return accepted_samples, accepted_samples_objective
        k = k + 1   
    return accepted_samples, accepted_samples_objective

# Main Program
P_samples = [10, 100, 1000]     # No. of Points
n_dim_lst = [i for i in range(1,10)]
size_domain = 1000
objective_value = 0
fig, axs = plt.subplots(1,1,figsize = (10,6))
axs.set_title(f'Random Sampling')
plt.subplots_adjust(hspace=0.5)
colors = ['red','blue','green','purple']
markers = ['o', 'v', 's', '+', 'x']
# iterate over P_samples where P_samples are changing
# Sub-Part (c)
# random sampling
for itr, num_samples in enumerate(P_samples):
    min_obj = []        # stores the min objectove value for each dimension
    # iterate over N dimensions where dimensiona are increasing
    # Sub-Part (a)
    for case, n_dimension in enumerate(n_dim_lst):
        accepted_samples, accepted_samples_objective = multiDimensional_sampling(num_samples, n_dimension, case, size_domain)
        min_obj.append(accepted_samples_objective[-1])
        print(f'Case {case+1}: Dimension is {n_dimension} with a reached minimum equal to {min(accepted_samples_objective)}')
    #ax = plt.subplot(1, len(P_samples), itr+1)      # plots a sub-plot for each P_samples value
    min_obj_rounded = [round(val, 3) for val in min_obj]
    axs.plot(n_dim_lst,min_obj_rounded,color=colors[itr], label='No. of samples '+str(num_samples), marker=markers[itr])
    axs.set_xlabel('N no. of dimensions', fontsize=8)
    axs.set_ylabel('min. objective value', fontsize=10)
    axs.legend(loc='best')
    for i, txt in enumerate(zip(n_dim_lst, min_obj_rounded)):
        axs.annotate(txt, (n_dim_lst[i], min_obj_rounded[i]), color=colors[itr])

# Sub-part (b)
# uniform sampling 
print(f'uniform sampling\n')
P_samples = [10, 20, 30]     # No. of Points
n_dim_lst = [i for i in range(1,5)]
fig, bxs = plt.subplots(1,1,figsize = (10,6))
bxs.set_title(f'Uniform Sampling')
colors = ['red','blue','green','purple']
markers = ['o', 'v', 's', '+', 'x']
r_min, r_max = -1.0, 1.0
for itr, num_samples in enumerate(P_samples):
    min_obj = []        # stores the min objective value for each dimension
    # iterate over N dimensions where dimensions are increasing
    # Sub-Part (a)
    for case, n_dimension in enumerate(n_dim_lst):
        accepted_samples_objective = []
        # Create a grid of points for each combination of dimensions 
        linspace = tuple(np.linspace(r_min, r_max, num_samples) for _ in range(n_dimension))
        # Create a grid of points for each combination of dimensions  
        inputs = np.array(list(itertools.product(*linspace)))
        for counter in range(len(inputs)):
            a_result = objective_vector(inputs[counter])
            accepted_samples_objective.append(a_result)
        min_obj.append(min(accepted_samples_objective))
    # sub-plot construction
    #ax = plt.subplot(1, len(P_samples), itr+1)      # plots a sub-plot for each P_samples value
    min_obj_rounded = [round(val, 3) for val in min_obj]
    bxs.plot(n_dim_lst,min_obj_rounded,color=colors[itr], label='No. of samples '+str(num_samples), marker=markers[itr])
    bxs.set_xlabel('N no. of dimensions', fontsize=8)
    bxs.set_ylabel('min. objective value', fontsize=10)
    bxs.legend(loc='best')
    #for i, txt in enumerate(zip(n_dim_lst, min_obj_rounded)):
    #    axs[itr].annotate(txt, (n_dim_lst[i], min_obj_rounded[i]))


# plotting 
plt.show()
