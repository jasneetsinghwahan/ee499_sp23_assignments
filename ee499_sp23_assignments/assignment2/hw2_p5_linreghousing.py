"""
this code uses gradient descent method on linear regression problem
gradient descent is calculated by hand for three loss functions i.e.
L1 and L2
This code has facilities to save the key ops from multiple runs of thus program
into a csv file
Graphs generated are also saved into a file along with timestamp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import os   # to move around files
import sys  # wanted to introduce some helper functions from different directories, but couldn't do so
import copy
import random       # to generate the seed number randomly

# for bookkeeping
import csv
from datetime import datetime

# custom modules - basically the functions shared in the jupyter notebook titled A2_hw2_datasets.ipynb
# done to keep the problem file small and specific to the problem at hand
import dataset_helper as dshelp 
import gradient_hand as gradhand

log_loc_assignment2 = os.path.abspath("./ee499_sp23_assignments/assignment2/")
data_loc_assignment2 = os.path.abspath("./ee499_sp23_assignments/data/hw2_data/")
graph_loc_assignment2 = os.path.abspath("./ee499_sp23_assignments/assignment2/graphs/")
logfilename = "hw2_problem5_housing.csv"
logfilename_completepath = os.path.join(log_loc_assignment2,logfilename)
current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
fn_time_string = time_string.replace(':', '_').replace('-', '_')

L1CLASSIFIER = "L1CLASSIFIER"
L2CLASSIFIER = "L2CLASSIFIER"
L2CLASSIFIERNORM = "L2CLASSIFIERNORM"
"""
load the 1st dataset, change the max_its
"""
data_file_name = 'housing.csv'
complete_data_file_name = os.path.join(data_loc_assignment2,data_file_name)
# this function modified to read CSV file and store as a pandas data frame
dataframe = pd.read_csv(complete_data_file_name)
column_names = dataframe.columns
print(f'original column ordering: {column_names}')

"""
data cleaning
"""
# reports column name was count of nans
nan_count = dataframe.isna().sum()
print(f'nan_count original dataframe: {nan_count}')
# population column had the nans reported
tbedroom_series = dataframe['total_bedrooms']
# take the average of all remaining elements
tbedroom_series_avg = tbedroom_series.mean(skipna=True)
# replace all elements where nan was reported with the average
tbedroom_series.fillna(value = tbedroom_series_avg, inplace=True)
# put the dataseries back into the original dataframe
dataframe['total_bedrooms'] = tbedroom_series
nan_count_up = dataframe.isna().sum()
print(f'nan_count after revision : {nan_count_up}')

# the 'ocean_proximity' doesn't contain numbers
# find unique labels in 'ocean_proximity'
ocean_proxmity_series = dataframe['ocean_proximity']
ops_unique_labels = ocean_proxmity_series.unique()
ops_unique_labels_count = ocean_proxmity_series.nunique()
mapping = {ops_unique_labels[0]: 1, ops_unique_labels[1]: 2,ops_unique_labels[2]: 3,ops_unique_labels[3]: 4,ops_unique_labels[4]: 5}
print(f'mapping: {mapping}')
# replace all text labels with the mapped numbers
ocean_proxmity_series_up = ocean_proxmity_series.replace(mapping)
ops_unique_labels_up = ocean_proxmity_series_up.unique()
# put the dataseries back into the original dataframe
dataframe['ocean_proximity'] = ocean_proxmity_series_up
#shuffle the data
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
# move the column titled 'median_house_value' to the extreme right 
median_house_value_series = dataframe['median_house_value']
dataframe['mmedian_house_value'] = median_house_value_series
formatted_dataframe = dataframe.drop('median_house_value',axis=1)
column_names_up = dataframe.columns
print(f'revised column ordering: {column_names_up}')

# move the column titled 
x_train_1 = formatted_dataframe.iloc[0:17540,:-1].values
x_test_1 = formatted_dataframe.iloc[17540:,:-1].values
y_train_1 = formatted_dataframe.iloc[0:17540,-1].values
y_test_1 = formatted_dataframe.iloc[17540:,-1].values

#assert x_train_1.shape[0] == y_train_1[0].shape, f'Shape mismatch between X and Y training'
#assert x_test_1.shape[0] == y_test_1[0].shape, f'Shape mismatch between X and Y test'
N_no_of_features_train_set1 = x_train_1.shape[1]

# Generate an array/list where each element is e_p = y_p - w^T.x_p where w^T is the best weight vector
# for both training and test data 
def error_func(w,x,y):
    return (y - gradhand.linmodel(x_p = x, w = w))

#------------------------------
# parameters for the run
#------------------------------
debug_print = False
if debug_print == True:
    np.random.seed(0) # set the seed to get the same random numbers every time
    random_seed = 0
else:
    random_seed = random.randint(0, 100000)
    print("Random seed: ", random_seed)
    np.random.seed(random_seed)
# +1 added to add bias
w = (10 * (2 * np.random.rand(N_no_of_features_train_set1 + 1) - 1) - 5)[:,np.newaxis]
alpha = 0.001; max_its = 10000 
current_classifier = "L1CLASSIFIER"
#current_classifier = "L2CLASSIFIER"
#current_classifier = "L2CLASSIFIERNORM"
#------------------------------
if (current_classifier == L1CLASSIFIER):
    weight_history_training, cost_history_training = gradhand.grad_l1loss(w = w, x = x_train_1, y = y_train_1, alpha = alpha, max_its = max_its, debug_print = debug_print)
    bestl1cost_train = cost_history_training[-1]
    bestl2cost_train = gradhand.least_squares(w = weight_history_training[-1], x = x_train_1, y = y_train_1)
    print(f"Best L2 loss function value on the training set for {current_classifier} classifier: {bestl2cost_train}")
    print(f"Best L1 loss function value on the training set for {current_classifier} classifier: {bestl1cost_train}")
    bestl2cost_test = gradhand.least_squares(w = weight_history_training[-1], x = x_test_1, y = y_test_1)
    print(f"Best L2 loss function value on the test set for {current_classifier} classifier: {bestl2cost_test}")
    bestl1cost_test = gradhand.l1_loss(w = weight_history_training[-1], x = x_test_1, y = y_test_1)
    print(f"Best L1 loss function value on the test set for {current_classifier} classifier: {bestl1cost_test}")
    e_err_train_vect = error_func(w = weight_history_training[-1],x = x_train_1, y = y_train_1).flatten()
    e_err_test_vect = error_func(w = weight_history_training[-1],x = x_test_1, y = y_test_1).flatten()
elif (current_classifier == L2CLASSIFIER):
    weight_history_training, cost_history_training = gradhand.grad_l2loss(w = w, x = x_train_1, y = y_train_1, alpha = alpha, max_its = max_its, debug_print = debug_print)
    bestl2cost_train = cost_history_training[-1]
    print(f"Best L2 loss function value on the training set for {current_classifier} classifier: {bestl2cost_train}")
    bestl1cost_train = gradhand.l1_loss(w = weight_history_training[-1], x = x_train_1, y = y_train_1)
    print(f"Best L1 loss function value on the training set for {current_classifier} classifier: {bestl1cost_train}")
    bestl2cost_test = gradhand.least_squares(w = weight_history_training[-1], x = x_test_1, y = y_test_1)
    print(f"Best L2 loss function value on the test set for {current_classifier} classifier: {bestl2cost_test}")
    bestl1cost_test = gradhand.l1_loss(w = weight_history_training[-1], x = x_test_1, y = y_test_1)
    print(f"Best L1 loss function value on the test set for {current_classifier} classifier: {bestl1cost_test}")
    e_err_train_vect = error_func(w = weight_history_training[-1],x = x_train_1, y = y_train_1).flatten()
    e_err_test_vect = error_func(w = weight_history_training[-1],x = x_test_1, y = y_test_1).flatten()
elif (current_classifier == L2CLASSIFIERNORM):
    weight_history_training, cost_history_training = gradhand.grad_l2loss_norm(w = w, x = x_train_1, y = y_train_1, alpha = alpha, max_its = max_its, debug_print = debug_print)
    bestl2cost_train = cost_history_training[-1]
    print(f"Best L2 loss function value on the training set for {current_classifier} classifier: {bestl2cost_train}")
    bestl1cost_train = gradhand.l1_loss(w = weight_history_training[-1], x = x_train_1, y = y_train_1)
    print(f"Best L1 loss function value on the training set for {current_classifier} classifier: {bestl1cost_train}")
    bestl2cost_test = gradhand.least_squares(w = weight_history_training[-1], x = x_test_1, y = y_test_1)
    print(f"Best L2 loss function value on the test set for {current_classifier} classifier: {bestl2cost_test}")
    bestl1cost_test = gradhand.l1_loss(w = weight_history_training[-1], x = x_test_1, y = y_test_1)
    print(f"Best L1 loss function value on the test set for {current_classifier} classifier: {bestl1cost_test}")
    e_err_train_vect = error_func(w = weight_history_training[-1],x = x_train_1, y = y_train_1).flatten()
    e_err_test_vect = error_func(w = weight_history_training[-1],x = x_test_1, y = y_test_1).flatten()


# record the results
print(f"Best weight vector for {current_classifier}: {weight_history_training[-1]}")
print(f"Shape of best weight vector for {current_classifier}: {weight_history_training[-1].shape}")if debug_print else print("",end="")

# learning curve for training
fig, (axs) = plt.subplots(1,1,figsize=(10, 5))
title_1 = rf'dataset {data_file_name} iterations {max_its} learning_rate {alpha} classifier {current_classifier}'
fig.suptitle(title_1)
fig.tight_layout()
axs.plot(range(0,max_its+1), cost_history_training, 'k-', linewidth=1)
axs.set_title(rf'$g(w^k)$ vs. step $k$ for training data {data_file_name}', fontsize=8)
axs.set_xlabel('step k', fontsize=12)
axs.set_ylabel(r'$g(w^k)$', fontsize=12)
# adds the annotation to the last point
x_last, y_last = max_its+1, cost_history_training[-1]
axs.annotate(f'({x_last}, {y_last})', (x_last, y_last), textcoords='offset points', xytext=(-15,10), ha='center', fontsize=8)
axs.plot(x_last, y_last, 'ro', markersize=10, label='Lowest cost')
# save the graph in a file also
shorttitle_1 = rf'{data_file_name}_iterations_{max_its}_learning_rate_{alpha}_classifier_{current_classifier}'
graphfile_name1 = shorttitle_1 + fn_time_string + ".jpg" 
graph_completefile_name1 = os.path.join(graph_loc_assignment2,graphfile_name1)
plt.savefig(graph_completefile_name1,format='jpg',bbox_inches='tight')
plt.show()

# histograms
## flatten is added so that we can convert the rows from shape (1,500) to (500,)
fig, (bx1, bx2) = plt.subplots(1,2)
fig.suptitle(title_1)
kwargs = dict(alpha = 0.5, bins = 100, edgecolor = 'black')
bx1.hist(e_err_train_vect, **kwargs, color ='g', label = 'class 1')
bx1.set_title(label = 'Training')
bx1.set_ylabel(ylabel='Frequency')
bx1.set_xlabel('error')
bx1.set_xlim(e_err_train_vect.min(), e_err_train_vect.max())
bx2.hist(e_err_test_vect, **kwargs, color ='g', label = 'class 1')
bx2.set_title(label = 'Test')
bx2.set_ylabel(ylabel='Frequency')
bx2.set_xlabel('error')
bx2.set_xlim(e_err_test_vect.min(), e_err_train_vect.max())
# save the graph
graphfile_train_name = shorttitle_1 + fn_time_string + "_histogram.jpg" 
graph_completefile_train_name = os.path.join(graph_loc_assignment2,graphfile_train_name)
plt.savefig(graph_completefile_train_name,format='jpg',bbox_inches='tight')
plt.show()

"""
log all the necesssary outputs to a csv file
"""
if not os.path.exists(logfilename_completepath):
    with open(logfilename_completepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time","dataset","iterations count","learning rate","classifier","best L2 loss training","best L1 Loss training","best L2 loss testing" ,"best L1 loss testing","random seed number","best weight vector"])
        writer.writerow([fn_time_string, data_file_name, max_its, alpha, current_classifier, bestl2cost_train, bestl1cost_train,bestl2cost_test,bestl1cost_test,random_seed,weight_history_training[-1].ravel()])
else:
    with open(logfilename_completepath, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([fn_time_string, data_file_name, max_its, alpha, current_classifier, bestl2cost_train, bestl1cost_train,bestl2cost_test,bestl1cost_test,random_seed,weight_history_training[-1].ravel()])

