"""
this code uses gradient dscent method on binary classifciation problem
gradient descent is clauclated by hand for three loss functions i.e.
L2, ReLU and Logistic Regression
This code has facilities to save the key ops from multiple runs of thus program
into a csv file
Graphs generated are also saved into a file along with timestamp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import os   # to move around files
import sys  # wanted to introduce some helper functions from different directories, but couldn't do so
import copy
import random

# for bookkeeping
import csv
from datetime import datetime
log_loc_assignment2 = os.path.abspath("./ee499_sp23_assignments/assignment2/")
data_loc_assignment2 = os.path.abspath("./ee499_sp23_assignments/data/hw2_data/")
graph_loc_assignment2 = os.path.abspath("./ee499_sp23_assignments/assignment2/graphs/")
logfilename = "hw2_problem4.csv"
logfilename_completepath = os.path.join(log_loc_assignment2,logfilename)
current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
fn_time_string = time_string.replace(':', '_').replace('-', '_')

# custom modules - basically the functions shared in the jupyter notebook titled A2_hw2_datasets.ipynb
# done to keep the problem file small and specific to the problem at hand
import dataset_helper as dshelp 
import gradient_hand as gradhand

# print debugging messages
debug_print = True

MSECLASSIFIER = "MSE"
PERCETRONRELUCLASSIFIER = "PerceptronReLU"
LOGISTICREGRESSIONCLASSIFIER = "LogisticRegression"
"""
load the 1st dataset, change the max_its
"""
#ip_file_1 = 'breast_cancer.npz'
ip_file_1 = 'classification_synthetic_1.npz'
input_file_1 = os.path.join(data_loc_assignment2,ip_file_1)
npzfile_1 = np.load(input_file_1)

# what are the arrays within the dataset
print(npzfile_1.files)

# load the dataset into individual arrays and print the shape of the arrays
x_train_1, x_test_1, y_train_1, y_test_1 = dshelp.load_data_npz(npzfile_1, labels_train = "labels_train", labels_test = "labels_test")
# check for NaN
np.unique(y_train_1, return_counts=True)

""" 
check what are the unique values in the y_train_1 and y_test_1
and if they are not 1 and -1 then convert them into 1 and -1 
"""
debug_print = False
unique_vals_train_1=np.unique(y_train_1)
unique_count_train_1= np.array([(y_train_1 == val).sum() for val in unique_vals_train_1])
# replaces all the values in the ndarray y_train_1 that previously had the value of 2 with -1
mod_y_train_1 = np.where(y_train_1 == 2,-1,y_train_1)
mod_unique_vals_train_1=np.unique(mod_y_train_1)
mod_unique_count_train_1= np.array([(mod_y_train_1 == val).sum() for val in mod_unique_vals_train_1])
# assert mod_unique_count_train_1[1] == unique_count_train_1[1], "modified labels count and original labels count don't match"
unique_vals_train_1=np.unique(y_test_1)
unique_count_train_1= np.array([(y_test_1 == val).sum() for val in unique_vals_train_1])
# replaces all the values in the ndarray y_test_1 that previously had the value of 2 with -1
mod_y_test_1 = np.where(y_test_1 == 2,-1,y_test_1)
mod_unique_vals_train_1=np.unique(mod_y_test_1)
mod_unique_count_test_1= np.array([(mod_y_test_1 == val).sum() for val in mod_unique_vals_train_1])
#assert mod_unique_count_test_1[mod_unique_vals_train_1[0]] == unique_count_train_1[unique_vals_train_1[0]], "modified labels count and original labels count don't match"

# merge arrays into creating a dataframe and also label the head of the columns, 
# plot_flag = true also plots the data
if (ip_file_1 == 'breast_cancer.npz'):
    df_train_1, df_test_1 = dshelp.create_dataframe_bc(x_train_1, x_test_1, mod_y_train_1, mod_y_test_1, plot_flag = False)
else:
    df_train_1, df_test_1 = dshelp.create_dataframe(x_train_1, x_test_1, mod_y_train_1, mod_y_test_1, plot_flag = False)
P_no_of_obs = df_train_1.shape[0]
N_no_of_features_train_set1 = df_train_1.shape[1]
N_no_of_features_test_set1 = df_test_1.shape[1]
print(f'head of dataset {df_train_1.head()}')
assert N_no_of_features_train_set1 == N_no_of_features_test_set1, f'training set and error set have different shapes'

"""
find the best weight vector and cost history for training data
"""
data_test_set1 = df_test_1.iloc[:,:-1].values
labels_test_set1 = df_test_1.iloc[:,-1].values
#------------------------------
# parameters for the run
#------------------------------
debug_print = False
if debug_print == True:
    random_seed = 22634
    np.random.seed(random_seed) # set the seed to get the same random numbers every time
else:
    random_seed = random.randint(0, 100000)
    print("Random seed: ", random_seed)
    np.random.seed(random_seed)
w = (10 * (2 * np.random.rand(N_no_of_features_train_set1) - 1) - 5)[:,np.newaxis]
alpha = 0.001; max_its = 10000 
#current_classifier = "MSE"
#current_classifier = "PerceptronReLU"
current_classifier = "LogisticRegression"
#------------------------------
if (current_classifier == MSECLASSIFIER):
    miscalculation_history_training_set1, weight_history_training_set1, cost_history_training_set1 = gradhand.grad_MSEloss_opt(w = w, given_data_frame = df_train_1, alpha = alpha, max_its = max_its, debug_print = debug_print)
    cost_test_set1 = gradhand.least_squares(weight_history_training_set1[-1], data_test_set1, labels_test_set1)
elif (current_classifier == PERCETRONRELUCLASSIFIER):
    miscalculation_history_training_set1, weight_history_training_set1, cost_history_training_set1 = gradhand.grad_preceptron_ReLuloss(w = w, given_data_frame = df_train_1, alpha = alpha, max_its = max_its, debug_print = debug_print)
    cost_test_set1 = gradhand.ReLU_cost(weight_history_training_set1[-1], data_test_set1, labels_test_set1)
elif (current_classifier == LOGISTICREGRESSIONCLASSIFIER):
    miscalculation_history_training_set1, weight_history_training_set1, cost_history_training_set1 = gradhand.grad_logloss(w = w, given_data_frame = df_train_1, alpha = alpha, max_its = max_its, debug_print = debug_print)
    cost_test_set1 = gradhand.logloss_cost(weight_history_training_set1[-1], data_test_set1, labels_test_set1)

print(f"Best weight vector for {current_classifier}: {weight_history_training_set1[-1]}")
print(f"Shape of best weight vector for {current_classifier}: {weight_history_training_set1[-1].shape}")if debug_print else print("",end="")
print(f"Best loss function value on the training set is {current_classifier}: {cost_history_training_set1[-1]}")
print(f"No. of miscalculations on the training set {current_classifier}: {miscalculation_history_training_set1[-1]}")
accuracy = 1-miscalculation_history_training_set1[-1]/float(P_no_of_obs)
print(f"Accuracy % on the training set {current_classifier}: {accuracy}")

"""
record the cost for testing set1
"""
print(f"Best loss function value on the testing set is: {cost_test_set1}")

"""
plot the learning rate for training data for Set 1
"""
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 5))
title_1 = rf'dataset {ip_file_1} iterations {max_its} learning_rate {alpha} classifier {current_classifier}'
fig.suptitle(title_1)
fig.tight_layout()
ax2.plot(range(0,max_its+1), cost_history_training_set1, 'k-', linewidth=1)
ax2.set_title(rf'$g(w^k)$ vs. step $k$ for training data {ip_file_1}', fontsize=8)
ax2.set_xlabel('step k', fontsize=12)
ax2.set_ylabel(r'$g(w^k)$', fontsize=12)
# adds the annotation to the last point
x_last, y_last = max_its+1, cost_history_training_set1[-1]
ax2.annotate(f'({x_last}, {y_last})', (x_last, y_last), textcoords='offset points', xytext=(-15,10), ha='center', fontsize=8)
ax2.plot(x_last, y_last, 'ro', markersize=10, label='Lowest cost')
# plot the miscalculation vs iterations for training data for Set 1
ax1.plot(range(0,max_its+1), miscalculation_history_training_set1, 'g-', linewidth=1)
ax1.set_title(rf'No. of miscalculations vs. step $k$ for training data {ip_file_1}', fontsize=8)
ax1.set_xlabel('step k', fontsize=12)
ax1.set_ylabel(r'No. of miscalculations', fontsize=12)
# adds the annotation to the last point
x_last, y_last = max_its+1, round(miscalculation_history_training_set1[-1],3)
ax1.annotate(f'({x_last}, {y_last})', (x_last, y_last), textcoords='offset points', xytext=(-15,10), ha='center', fontsize=8)
ax1.plot(x_last, y_last, 'ro', markersize=10, label='Lowest classification')
# save the graph in a file also
shorttitle_1 = rf'{ip_file_1}_iterations_{max_its}_learning_rate_{alpha}_classifier_{current_classifier}'
graphfile_name1 = shorttitle_1 + fn_time_string + ".jpg" 
graph_completefile_name1 = os.path.join(graph_loc_assignment2,graphfile_name1)
plt.savefig(graph_completefile_name1,format='jpg',bbox_inches='tight')
plt.show()


"""
build the scatter plot and plot the decision boundary
"""
# Calculate the min and max values of each column
if (ip_file_1 == 'breast_cancer.npz'):
    x_min_value = df_train_1.min().min()
    x_max_value = df_train_1.max().max()
    y_min_value = df_train_1.min().min()
    y_max_value = df_train_1.max().max()
else:   
    x_min_value = df_train_1['X1'].min()
    x_max_value = df_train_1['X1'].max()
    y_min_value = df_train_1['X2'].min()
    y_max_value = df_train_1['X2'].max()
minx = min(x_min_value,y_min_value)
maxx = max(x_max_value,y_max_value)
x_min = minx-0.1; x_max = maxx+0.1
col = np.where(df_train_1["Label"]==1,'r','b')
plt.figure(figsize=(7, 9))
plt.scatter(df_train_1["X1"],df_train_1["X2"],s=100,c=col, edgecolors='k')
plt.xlabel('X1', fontsize = 12)
plt.ylabel('X2', fontsize = 12)
title_2 = f'Linear Binary Classification dataset {ip_file_1} iterations {max_its} learning_rate {alpha} classifier {current_classifier}'
plt.title(title_2)
#plt.title('Classification Problem', fontsize=16)
r = np.linspace(x_min,x_max,1000)
best_w = weight_history_training_set1[-1]
z = - best_w[0]/best_w[2] - best_w[1]/best_w[2]*r
plt.plot(r,z,linewidth = 2,zorder = 3)
plt.plot(r,z,linewidth = 2.75,color = 'k',zorder = 2)
# save the file
shorttitle_2 = f'Classification_{ip_file_1}_iterations_{max_its}_learning_rate_{alpha}_classifier_{current_classifier}'
graphfile_name2 = shorttitle_2 + fn_time_string + "_scatterplot.jpg" 
graph_completefile_name2 = os.path.join(graph_loc_assignment2,graphfile_name2)
plt.savefig(graph_completefile_name2,format='jpg',bbox_inches='tight')
plt.show()

"""
build the histogram displaying the distance from the boufor the loss function for each data point 
"""
# Step 1: Generate an array/list where each element is x_p.w/A where A = eucledian distance of \
# best weight vector
def lineardistance_vect(x_p,w):
    # compute linear combination and return
    a = (w[0] + np.dot(x_p,w[1:]))/np.sqrt(np.dot(w.T,w))
    return a.T

# calculate distances separately for each class
# df_train_1.iloc[:,-1] == 1 selects the entire row of the data frame that has \
# corresponding element in the last column == 1
# and then outer df_train_1().iloc[:,:-1].values selects the all such rows ecluding the last column
class_1_dataframe = (df_train_1[df_train_1.iloc[:,-1] == 1]).iloc[:,:-1].values
class_minus1_dataframe = (df_train_1[df_train_1.iloc[:,-1] == -1]).iloc[:,:-1].values
# flatten is added so that we can convert the rows from shape (1,500) to (500,)
class_1_distances = lineardistance_vect(x_p = class_1_dataframe, w = weight_history_training_set1[-1]).flatten()
class_minus1_distances = lineardistance_vect(x_p = class_minus1_dataframe, w = weight_history_training_set1[-1]).flatten()
kwargs = dict(alpha = 0.5, bins = 100)
plt.hist(class_1_distances, **kwargs, color ='g', label = 'class 1')
plt.hist(class_minus1_distances, **kwargs, color ='b', label = 'class 2')
plt.gca().set(title = title_2, ylabel ='Frequency')
plt.xlim(min(class_1_distances.min(), class_minus1_distances.min()), max(class_1_distances.max(), class_minus1_distances.max()))
# save the graph
graphfile_name3 = shorttitle_2 + fn_time_string + "_histogram.jpg" 
graph_completefile_name3 = os.path.join(graph_loc_assignment2,graphfile_name3)
plt.savefig(graph_completefile_name3,format='jpg',bbox_inches='tight')
plt.show()

"""
log all the necesssary outputs to a csv file
"""
if not os.path.exists(logfilename_completepath):
    with open(logfilename_completepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time","dataset","iterations count","learning rate","classifier","best loss function value on training","best loss function value on testing","no. of miscalculations on training set", "accuracy %","random seed number","best weight vector"])
        writer.writerow([fn_time_string,ip_file_1,max_its,alpha,current_classifier,cost_history_training_set1[-1],cost_test_set1,miscalculation_history_training_set1[-1], accuracy, random_seed,weight_history_training_set1[-1].ravel()])
else:
    with open(logfilename_completepath, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([fn_time_string,ip_file_1,max_its,alpha,current_classifier,cost_history_training_set1[-1],cost_test_set1,miscalculation_history_training_set1[-1], accuracy, random_seed,weight_history_training_set1[-1].ravel()])
