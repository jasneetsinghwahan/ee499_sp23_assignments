# Homework 1
# Problem 1
# Using Global Optimization Approach
# P <- no. of points in each dimension
# N <- No. of Dimensions
# w <- vector in the parameter space or weight space
# M <- No. of evaluations requried 
# I jave done this exercise over P=2,3,4 (Part 1) and N=2,3,4 (part 2) \
# as execution time was increasing a lot


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

# g(w)
# input: vector quantity in N dimensions
# output: scaler
# defining the function g(w) that accepts vectors and does 
# operation on vectors 
#def objective_vector(w):
#	return np.dot(w, w.T) + 0.2
def objective_vector(w, case,sub_part):
    M_evaluations_lst[sub_part][case] += 1
    return np.dot(w, w.T) + 0.2

# function where evaluation takes place
# output is accepted_samples and accepted_sample_objectives 
def multiDimensional_sampling(number_samples, n_dimension, case, size_domain, sub_part):
    X = np.linspace(r_min, r_max, size_domain)
    accepted_samples = []
    accepted_samples_objective = []
    minimum = 1e30
    k = 0
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
        if objective_vector(sample, case, sub_part) < minimum:
            stuck_flag = 0
            accepted_samples.append(sample)
            minimum = objective_vector(sample,case, sub_part)
            accepted_samples_objective.append(minimum)
            if abs(minimum - objective_value) <= 1e-2:
                print(f"Case {case+1} convergence is reached with {k+1} samples")
                return accepted_samples, accepted_samples_objective
        k = k + 1   
    return accepted_samples, accepted_samples_objective

# Main Program
P_points_lst = [2,3,4]
n_dimension_lst = 3 
# Expression tying No. of evaluations made with Dimensions and number of points is
# M = P^N
P_samples_lst = [x**n_dimension_lst for x in P_points_lst]
objective_value = 0.2
r_min = 0; r_max = 1
size_domain = 100
M_evaluations_lst = np.zeros([2, len(P_points_lst)])
#run_times = np.zeros(len(n_dimension_lst))

# Part 1
# following loop would run over P_samples_lst and save the evaluations made in a list M_evaluations_lst 
for case, n_samples in enumerate(P_samples_lst):
    sub_part = 0
    #tic = time.time()
    accepted_samples, accepted_samples_objective = multiDimensional_sampling(n_samples, n_dimension_lst, case, \
        size_domain,sub_part)
    #toc = time.time()
    print(f'Case {case+1}: No. of Samples are {n_samples} with a reached minimum equal to {min(accepted_samples_objective)} \
        and no. of evaluations {M_evaluations_lst[sub_part][case]}')
    #run_times[case] = 1000 * (toc - tic)
    #print(f"Duration of Case {case+1}: {run_times[case] : .4f} ms \n")

# Part 2
# following loop would record M_evaluations_lst when N no. of dimensions changes 
n_dimension_lst2 = [2,3, 4] 
P_fixed = 10
P_samples_lst1 = [P_fixed**x for x in n_dimension_lst2]
sub_part = 1
for case, n_samples in enumerate(n_dimension_lst2):
    #tic = time.time()
    accepted_samples, accepted_samples_objective = multiDimensional_sampling(P_samples_lst1[case], n_samples , case, \
        size_domain,sub_part)
    #toc = time.time()
    print(f'Case {case+1}: No. of dimensions are {n_samples} with a reached minimum equal to {min(accepted_samples_objective)} \
        and no. of evaluations {M_evaluations_lst[sub_part][case]}')

# plotting two plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
ax1.set_title("For Fixed N = 3")
ax1.plot(P_samples_lst, M_evaluations_lst[0], 'k')
ax1.scatter(P_samples_lst, M_evaluations_lst[0]) 
ax1.set_yscale('log')
ax1.set_xlabel('P No. of samples', fontsize=12)
ax1.set_ylabel('M No. of evaluations (log)', fontsize=12)
ax2.set_title("For Fixed P = 10")
ax2.set_yscale('log')
ax2.plot(n_dimension_lst2 , M_evaluations_lst[1], 'k')
ax2.scatter(n_dimension_lst2 , M_evaluations_lst[1]) 
ax2.set_xlabel('N No. of dimensions', fontsize=12)
ax2.set_ylabel('M No. of evaluations (log)', fontsize=12)
for i, txt in enumerate(zip(n_dimension_lst2, M_evaluations_lst[1])):
    ax1.annotate(txt, (n_dimension_lst2[i], M_evaluations_lst[1][i]))
plt.show()
