# Problem 2.3 Watt, et. al.:

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

# objective_function
g = lambda w: math.tanh(4 * w[0] + 4 * w[1]) + max(0.4 * w[0]**2, 1) + 1

# random search algo.
def random_search(g, alpha_choice, max_its, w, num_samples):
    # run random search
    weight_history = []         # container for weight history
    cost_history = []           # container for corresponding cost function history
    alpha = 0
    for k in range(1, max_its + 1):        
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1 / k
        else:
            alpha = alpha_choice
            
        # record weights and cost evaluation
        weight_history.append(w)
        cost_history.append(g(w))
        
        # construct set of random unit directions
        directions = np.random.randn(num_samples, np.size(w))
        # norms = np.sqrt(np.sum(directions * directions, axis = 1))[:, np.newaxis]
        norms = np.linalg.norm(directions, axis = 1)[:, np.newaxis]
        directions = directions / norms   
        
        ### pick best descent direction
        # compute all new candidate points
        w_candidates = w + alpha * directions
        
        # evaluate all candidates
        evals = np.array([g(w_val) for w_val in w_candidates])

        # if we find a real descent direction take the step in its direction
        ind = np.argmin(evals)
        if g(w_candidates[ind]) < g(w):
            # pluck out best descent direction
            d = directions[ind,:]
        
            # take step
            w = w + alpha * d
        
    # record weights and cost evaluation
    weight_history.append(w)
    cost_history.append(g(w))
    return weight_history,cost_history

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

# run random search algorithm 
alpha_choice = 1; w = np.asarray([2, 2]); num_samples = 1000; max_its = 8;
weight_history_1,cost_history_1 = random_search(g, alpha_choice, max_its, w, num_samples)
plot_contours(g, weight_history_1, view=[20,100],flag3D=True, title = 'Random Search')