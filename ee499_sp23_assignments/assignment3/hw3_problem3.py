"""
Note FOR EVALUATION: You have to specify your path in order to locate the 'house-votes-84.csv' file
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import copy
from datetime import datetime       # for naming the graphs with timestamps
import getpass                      # to get the username
import csv                          # for logs

#------------------------------
# file structure
#------------------------------
username = getpass.getuser()        # Get the current username
current_dir = os.getcwd()
log_fname = "hw3_problem3.csv"
data_fname = 'house-votes-84.csv'
data_rel_path = 'ee499_ml_spring23/hw_helpers/HW3/'
log_rel_path = 'ee499_sp23_assignments/assignment3/'
img_rel_path = 'ee499_sp23_assignments/assignment3/'
data_dir_abs_path = os.path.join(current_dir,data_rel_path)
log_dir_abs_path = os.path.join(current_dir,log_rel_path)
img_dir_inter_abs_path = os.path.join(current_dir,img_rel_path)
if username == "jasne":
    img_dir_name = "img"
    if not os.path.exists(os.path.join(img_dir_inter_abs_path, img_dir_name)):
        os.makedirs(os.path.join(img_dir_inter_abs_path, img_dir_name))
    img_dir_abs_path = os.path.join(img_dir_inter_abs_path,img_dir_name)
    log_fname_abs_path = os.path.join(log_dir_abs_path,log_fname)
else:
    img_dir_name = "img_USC2398388668"
    current_dir = os.getcwd()
    img_dir_abs_path = os.path.join(current_dir,img_dir_name)
current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
fn_time_string = time_string.replace(':', '_').replace('-', '_')


#------------------------------
# programming parameters
#------------------------------
DEVELOPMENT = False          # True means that this program is being developed, false means this program can be deployed

#load data
# x.shape = (435,16)
x = np.genfromtxt(data_dir_abs_path + 'house-votes-84.csv', delimiter=',', skip_header=True)

#You will write some parts related to clustering. You need to complete 5 Tasks.
#This is a step by step guide to run kmeans algorithm from scratch
def kmeans_fit(x, N_clusters, N_iterations, num_init, epsilon=1e-3):
    P, N = x.shape
    distances = np.zeros((P, N_clusters))
    loss_best = 1e10
    cluster_heads_best = np.zeros((N_clusters, N))
    cluster_assignments = np.zeros(P, dtype=int)
    values = [-1.0,1.0]
    cluster_shape = (N,)
    n = 0
    for n in range(num_init):

        ### TASK 1: sample "cluster_heads" from a uniform distribution [-1,1]. Make sure the size of "cluster_heads" is specified correctly
        cluster_heads = np.random.choice(values, size =(N_clusters,) + cluster_shape)

        not_done = True
        last_loss = 1e10
        iter_counter = 0
        # this while loop iterates over the N_iterations
        while not_done and iter_counter < N_iterations:
            
            ### TASK 2: Calculate the Euclidean distance "distances" from every point in the data to every "cluster_heads".
            for i in range(N_clusters):
                distances[:,i] = np.sqrt(np.sum((x-cluster_heads[i])**2,axis=1))

            ### TASK 3: "cluster_assignments" calculation. Hint: How do you assign points to clusters?
            cluster_assignments = np.argmin(distances, axis=1)

            ### TASK 4: compute the loss. You need to loop over N_clusters and evaluate the total distance of points in a cluster
            loss = 0.0
            i = 0
            for i in range(N_clusters):
                mask = (cluster_assignments == i)
                loss  += np.sum(np.linalg.norm(x[mask] - cluster_heads[i], axis=1))
            loss = loss / P

            ### check if we are done
            loss_change_fractional = (last_loss - loss) / loss

            # print(f'iteration = {i},  loss = {loss : 3.2e} last_loss = {last_loss : 3.2e} frac loss-delta: {loss_change_fractional : 3.2e}')
            if loss_change_fractional < epsilon:
                not_done = False
                print(f'this initialiation done: iteration = {iter_counter}, fractional loss = {loss_change_fractional : 3.2e}')
            else:
                iter_counter += 1
                last_loss = loss
                ### TASK 5: compute new "cluster_heads". You need to loop over N_clusters and evaluate the new centroid location
                for k in range(N_clusters):
                    # collect indices of points belonging to kth cluster
                    mask = (cluster_assignments == k)
                    if np.sum(mask) > 0:
                        cluster_heads[k,:] = np.mean(x[mask],axis=0, keepdims=True)
                    else:  # empty cluster
                        cluster_heads[k,:] = copy.deepcopy(cluster_heads[k,:])[:,np.newaxis].T
        # End of While loop for N_iterations
        
        # update the heads, assignments and loss at the end of all iterations for a specific initialization
        # provided that this specific initialization resulted in loss less than loss from all the previous initializations
        if loss < loss_best:
            cluster_heads_best = cluster_heads
            cluster_assignments_best = cluster_assignments
            loss_best = loss
            # print(f'n = {n}, new best loss: {loss_best}')
    # end of all initializations
    return cluster_heads_best, cluster_assignments_best, loss_best 

#------------------------------
# program begins here 
#------------------------------
N_clusters = np.arange(1, 5)
loss_values = []
cluster_heads = []
cluster_assignments = []

if DEVELOPMENT:
    random_seed = 0
    np.random.seed(random_seed) # set the seed to get the same random numbers every time
else:
    random_seed = random.randint(0, 100000)
    np.random.seed(random_seed)
print("Random seed: ", random_seed)

for num_clusters in N_clusters:
    print(f'\nNumber of Clusters = {num_clusters}')
    heads, assigments, loss = kmeans_fit(x, num_clusters, N_iterations= 100, num_init= 5)
    loss_values.append(loss)
    cluster_heads.append(heads)
    cluster_assignments.append(assigments)

    plt.figure()
    for m in range(num_clusters):
        plt.plot(np.arange(1,17), np.mean(x[assigments == m], axis = 0), linestyle='--', marker='o', label=f'votes cluster {m}')
    plt.grid(':')
    plt.legend()
    plt.xlabel('initiative')
    plt.ylabel('average vote for cluster')
    # save the graph
    fig_fname = fn_time_string + "_" + str(num_clusters) + "_clusters_issues_vs_avgvoteforcluster.jpg" 
    fig_fname_abs_path = os.path.join(img_dir_abs_path,fig_fname)
    plt.savefig(fig_fname_abs_path,format='jpg',bbox_inches='tight')
    plt.show()
    """log all required outputs"""
    unique, counts = np.unique(cluster_assignments[-1], return_counts=True)
    if username == "jasne":
        if not os.path.exists(log_fname_abs_path):    
            with open(log_fname_abs_path, 'w', newline='') as file:
                writer = csv.writer(file)
                if num_clusters == 1:
                    writer.writerow(["time","No. of clusters","loss", "random_seed", "Cluster0 count"])
                    writer.writerow([fn_time_string, num_clusters, loss_values[-1],random_seed, counts[0]])
                if num_clusters == 2:
                    writer.writerow(["time","No. of clusters","loss", "random_seed", "Cluster0 count","Cluster1 count"])
                    writer.writerow([fn_time_string, num_clusters, loss_values[-1],random_seed, counts[0], counts[1]])
                if num_clusters == 3:
                    writer.writerow(["time","No. of clusters","loss", "random_seed", "Cluster0 count","Cluster1 count", "Cluster2 count"])
                    writer.writerow([fn_time_string, num_clusters, loss_values[-1],random_seed, counts[0], counts[1], counts[2]])
                if num_clusters == 4:
                    writer.writerow(["time","No. of clusters","loss", "random_seed", "Cluster0 count","Cluster1 count", "Cluster2 count", "Cluster3 count"])
                    writer.writerow([fn_time_string, num_clusters, loss_values[-1],random_seed, counts[0], counts[1], counts[2], counts[3]])
        else:
            with open(log_fname_abs_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if num_clusters == 1:
                    writer.writerow([fn_time_string, num_clusters, loss_values[-1],random_seed, counts[0]])
                if num_clusters == 2:
                    writer.writerow([fn_time_string, num_clusters, loss_values[-1],random_seed, counts[0], counts[1]])
                if num_clusters == 3:
                    writer.writerow([fn_time_string, num_clusters, loss_values[-1],random_seed, counts[0], counts[1], counts[2]])
                if num_clusters == 4:
                    writer.writerow([fn_time_string, num_clusters, loss_values[-1],random_seed, counts[0], counts[1], counts[2], counts[3]])

plt.figure()
plt.plot(N_clusters, loss_values, color='b', linestyle='--', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('K-means Loss per Data Point')
plt.grid(':')
# save the graph
fig_fname = fn_time_string + "_no_of_clusters_vs_loss.jpg" 
fig_fname_abs_path = os.path.join(img_dir_abs_path,fig_fname)
plt.savefig(fig_fname_abs_path,format='jpg',bbox_inches='tight')
plt.show()