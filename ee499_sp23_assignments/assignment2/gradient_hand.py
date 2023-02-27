import numpy as np
from autograd import numpy as np
from autograd import value_and_grad
import time

## compute linear combination of input point
# return is scaler so transpose or no transpose doesn't matter
# shape returned when matrix is passed is of the form (1,P_No_of_obs)
def linmodel(x_p,w):
    # compute linear combination and return
    a = w[0] + np.dot(x_p,w[1:])
    return a.T

## a least squares function for linear regression
def least_squares(w,x,y):    
    cost = np.sum((linmodel(x,w)-y)**2)
    return cost/float(y.size)

## a least squares function for linear regression
def l1_loss(w,x,y):    
    cost = np.sum(np.abs(linmodel(x,w)-y))
    return cost/float(y.size)

## ReLU cost function for Perceptron learning
def ReLU_cost(w,x,y):
    cost = np.sum(np.maximum(0,-1 * y.reshape(x.shape[0],1) * linmodel(x, w).T)) 
    return cost/float(x.shape[0])

## Logistics regression applicable when labels belongs to (1,-1)
# shape returned when matrix is passed is of the form (1,P_No_of_obs)
def logloss_cost(w,x,y):
    cost = np.sum(np.log(1+np.exp(-1 * y.reshape(x.shape[0],1) * linmodel(x, w).T)))
    return cost/float(x.shape[0])

"""given a weight vector, iterate over all the data points and return accuracy"""
def accuracy_checker(model, w, given_data_frame, debug_print):
    data = given_data_frame.iloc[:,:-1].values
    labels = given_data_frame.iloc[:,-1].values
    P_no_of_obs = given_data_frame.shape[0] 
    num_mismatch = 0
    # function(data,w) evaluates the value
    sign_calculator = np.sign(model(data,w)).reshape(P_no_of_obs,)
    mismatch = [1 if a!= b else 0 for a,b in zip(sign_calculator,labels)] 
    num_mismatch = sum(mismatch)
    return num_mismatch

"""
input - weight_vector, pandas_dataframe, alpha (learning rate), max iterations and debug_print
output - computed weight vector
Least Sqaure Classifier
w includes bias term 
REMEMBER - always define applicable cost function and appplicable model variables
TBD - Logic that could terminate the for loop if combination of cost function\
      and accuracy is wobbling and/or triviallly incrementing
"""
def grad_MSEloss_opt(w, given_data_frame, alpha, max_its,debug_print):
    #"""verify the result with grad function"""
    applicable_cost_func = least_squares
    applicable_model = linmodel
    #gradient = value_and_grad(g) 
    data = given_data_frame.iloc[:,:-1].values
    labels = given_data_frame.iloc[:,-1].values
    #cost_eval,grad_eval = gradient(w,data,labels)
    cost_history = []        # container for corresponding cost function history
    weight_history = []      # container for weight history
    miscalculation_history = []
    # update with initial values
    miscalculation_history.append(accuracy_checker(model = applicable_model, w = w,given_data_frame=given_data_frame,debug_print=False))
    cost_history.append(applicable_cost_func(w, data, labels))
    weight_history.append(w)
    P_no_of_obs = given_data_frame.shape[0]
    """
    loop over iterations
    """
    for k in range(1,max_its+1):
        """
        calculate the error vector over all the observations 
        """
        dot_product_vect = linmodel(x_p = data, w = w)        
        e_error_vect = dot_product_vect - labels
        one_element_array_vect = np.ones((P_no_of_obs, 1))
        x_p_aug_vect = np.concatenate((one_element_array_vect,data),axis=1)
        # tranpose to ensure that w and grad_eval are of the same shape
        g_p_vect = e_error_vect * x_p_aug_vect.T
        grad_eval = 2/P_no_of_obs*np.sum(g_p_vect,axis = 1)
        w = w - alpha*grad_eval[:,np.newaxis]
        miscalculation_history.append(accuracy_checker(model = applicable_model, w = w,given_data_frame=given_data_frame,debug_print=False))
        cost_history.append(applicable_cost_func(w, data, labels))
        weight_history.append(w)
    return miscalculation_history,weight_history,cost_history

"""
input - weight_vector, pandas_dataframe, alpha (learning rate), max iterations and debug_print
output - computed weight vector
Perceptrron Classifier
w includes bias term 
REMEMBER - always define applicable cost function and appplicable model variables
TBD - Logic that could terminate the for loop if combination of cost function\
      and accuracy is wobbling and/or triviallly incrementing
Step 2 assumed linmodal which may have to be revisited later
"""
def grad_preceptron_ReLuloss(w, given_data_frame, alpha, max_its,debug_print):
    applicable_model = linmodel
    applicable_cost_function = ReLU_cost
    data = given_data_frame.iloc[:,:-1].values
    labels = given_data_frame.iloc[:,-1].values
    P_no_of_obs = given_data_frame.shape[0]
    #g = ReLU_cost
    #gradient = value_and_grad(g) 
    #cost_eval,grad_eval = gradient(w,data,labels,P_no_of_obs)
    cost_history = []        # container for corresponding cost function history
    weight_history = []      # container for weight history
    miscalculation_history = []
    # update with initial values
    miscalculation_history.append(accuracy_checker(model = applicable_model, w = w,given_data_frame=given_data_frame,debug_print=False))
    cost_history.append(applicable_cost_function(w = w, x = data, y = labels))
    weight_history.append(w)
    N_no_of_features = data.shape[1]
    for k in range(1,max_its+1):
        """
        calculate the error vector over all the observations 
        """
        # Step 1: Form the augment vector for x_p
        one_element_array_vect = np.ones((P_no_of_obs, 1))
        x_p_aug_vect = np.concatenate((one_element_array_vect,data),axis=1)
        # Step 2: calculate the product of -y_p * w * X_p in vectorized form
        eval_vect = -1 * labels.reshape(P_no_of_obs,1) * linmodel(x_p = data, w = w).T 
        # Step 3: for each row that evaluates to > 0, store the -y_p * x_p in vectorized form
        # CHECK THIS STEP AGAIN WHEN INDEX IS BETWEEN 499 and 502 and ONCE BETWEEN INDEX 504 and 507
        g_p_vect = [labels[i] * x_p_aug_vect[i] if eval_vect[i] > 0 else np.zeros((N_no_of_features+1,)) for i in range(len(eval_vect))]
        # g_p_vect os of the shape P rows and (N+1) columns
        g_p_vect = np.array(g_p_vect).reshape(P_no_of_obs,N_no_of_features+1)
        # Step 4: Sum each column of these P * (N+1) rows where P is no. of observations and \
        #           N are the no. of features
        grad_eval = -1 * 1/P_no_of_obs*np.sum(g_p_vect,axis = 0)
        # Step 5: Update w, cost_function and accuracy and go back to step 1
        # Shape of w is (N+1,1)
        w = w - alpha*grad_eval[:,np.newaxis]
        miscalculation_history.append(accuracy_checker(model = applicable_model, w = w,given_data_frame=given_data_frame,debug_print=False))
        cost_history.append(applicable_cost_function(w, data, labels))
        weight_history.append(w)
    return miscalculation_history,weight_history,cost_history

"""
input - weight_vector, pandas_dataframe, alpha (learning rate), max iterations and debug_print
output - computed weight vector
Logistic Regression Classifier
w includes bias term 
REMEMBER - always define applicable cost function and appplicable model variables
TBD - Logic that could terminate the for loop if combination of cost function\
      and accuracy is wobbling and/or triviallly incrementing
"""
def grad_logloss(w, given_data_frame, alpha, max_its,debug_print):
    applicable_model = linmodel
    applicable_cost_function = logloss_cost
    data = given_data_frame.iloc[:,:-1].values
    labels = given_data_frame.iloc[:,-1].values
    P_no_of_obs = given_data_frame.shape[0]
    #g = ReLU_cost
    #gradient = value_and_grad(g) 
    #cost_eval,grad_eval = gradient(w,data,labels,P_no_of_obs)
    cost_history = []        # container for corresponding cost function history
    weight_history = []      # container for weight history
    miscalculation_history = []
    # update with initial values
    miscalculation_history.append(accuracy_checker(model = applicable_model, w = w,given_data_frame=given_data_frame,debug_print=False))
    cost_history.append(applicable_cost_function(w = w, x = data, y = labels))
    weight_history.append(w)
    N_no_of_features = data.shape[1]
    for k in range(1,max_its+1):
        """
        calculate the error vector over all the observations 
        """
        # Step 1: Form the augment vector for x_p
        one_element_array_vect = np.ones((P_no_of_obs, 1))
        x_p_aug_vect = np.concatenate((one_element_array_vect,data),axis=1)
        # Step 2: form a matrix with rows = P observations and columns equal to augmented vector x_p
        denominator = 1+np.exp(labels.reshape(data.shape[0],1) * applicable_model(data, w).T)
        # this exotic operation is done so that we can multiply y of shape (P_No_of_obs,) with \
        # x_aug_vect of shape (P_No_of_obs * (N+1)) such that each row element of y multiples to\
        # corresponding row of x_p_aug_vect
        numerator = (labels * x_p_aug_vect.T).T
        eval_vect = numerator * (1/denominator)
        # Step 3: Sum each column of these P * (N+1) rows where P is no. of observations and \
        # g_p_vect is of the shape P rows and (N+1) columns
        grad_eval = -1 * 1/P_no_of_obs*np.sum(eval_vect,axis = 0)
        # Step 5: Update w, cost_function and accuracy and go back to step 1
        # Shape of w is (N+1,1)
        w = w - alpha*grad_eval[:,np.newaxis]
        miscalculation_history.append(accuracy_checker(model = applicable_model, w = w,given_data_frame=given_data_frame,debug_print=False))
        cost_history.append(applicable_cost_function(w, data, labels))
        weight_history.append(w)
    return miscalculation_history,weight_history,cost_history

"""
grad_l2loss
input - weight_vector, x array of shape (P_No_of_obs,N), y array of shape (P_No_of_obs,1),
alpha (learning rate), max iterations and debug_print
output - weight vector history and loss function history 
w includes bias term 
REMEMBER - always define applicable cost function and appplicable model variables
TBD - Logic that could terminate the for loop if combination of cost function\
      and accuracy is wobbling and/or triviallly incrementing
"""
def grad_l2loss(w, x, y, alpha, max_its, debug_print):
    weight_history = []
    cost_history = []
    applicable_cost_func = least_squares
    applicable_model = linmodel
    cost_history.append(applicable_cost_func(w, x, y))
    weight_history.append(w)
    P_no_of_obs = x.shape[0]
    """
    loop over iterations
    """
    for k in range(1,max_its+1):
        """
        calculate the error vector over all the observations 
        """
        e_error_vect = (applicable_model(x,w) - y)
        one_element_array_vect = np.ones((P_no_of_obs, 1))
        x_p_aug_vect = np.concatenate((one_element_array_vect,x),axis=1)
        # tranpose to ensure that w and grad_eval are of the same shape
        g_p_vect = e_error_vect * x_p_aug_vect.T
        grad_eval = 2/P_no_of_obs*np.sum(g_p_vect,axis = 1)
        w = w - alpha*grad_eval[:,np.newaxis]
        cost_history.append(applicable_cost_func(w, x, y))
        weight_history.append(w)
    return weight_history,cost_history

"""
grad_l1loss
input - weight_vector, x array of shape (P_No_of_obs,N), y array of shape (P_No_of_obs,1),
alpha (learning rate), max iterations and debug_print
output - weight vector history and loss function history 
w includes bias term 
REMEMBER - always define applicable cost function and appplicable model variables
TBD - Logic that could terminate the for loop if combination of cost function\
      and accuracy is wobbling and/or triviallly incrementing
"""
def grad_l1loss(w, x, y, alpha, max_its, debug_print):
    weight_history = []
    cost_history = []
    applicable_cost_func = l1_loss
    applicable_model = linmodel
    cost_history.append(applicable_cost_func(w, x, y))
    weight_history.append(w)
    P_no_of_obs = x.shape[0]
    """
    loop over iterations
    """
    for k in range(1,max_its+1):
        """
        calculate the error vector over all the observations 
        """
        e_error_vect_sign = np.sign(applicable_model(x,w) - y)
        one_element_array_vect = np.ones((P_no_of_obs, 1))
        x_p_aug_vect = np.concatenate((one_element_array_vect,x),axis=1)
        # tranpose to ensure that w and grad_eval are of the same shape
        g_p_vect = e_error_vect_sign * x_p_aug_vect.T
        grad_eval = 1/P_no_of_obs*np.sum(g_p_vect,axis = 1)
        w = w - alpha*grad_eval[:,np.newaxis]
        cost_history.append(applicable_cost_func(w, x, y))
        weight_history.append(w)
    return weight_history,cost_history


"""
grad_l2loss_norm
input - weight_vector, x array of shape (P_No_of_obs,N), y array of shape (P_No_of_obs,1),
alpha (learning rate), max iterations and debug_print
output - weight vector history and loss function history 
w includes bias term 
REMEMBER - always define applicable cost function and appplicable model variables
TBD - Logic that could terminate the for loop if combination of cost function\
      and accuracy is wobbling and/or triviallly incrementing
"""
def grad_l2loss_norm(w, x, y, alpha, max_its, debug_print):
    weight_history = []
    cost_history = []
    applicable_cost_func = least_squares
    applicable_model = linmodel
    cost_history.append(applicable_cost_func(w, x, y))
    weight_history.append(w)
    P_no_of_obs = x.shape[0]
    """
    loop over iterations
    """
    for k in range(1,max_its+1):
        """
        calculate the error vector over all the observations 
        """
        e_error_vect = (applicable_model(x,w) - y)
        one_element_array_vect = np.ones((P_no_of_obs, 1))
        x_p_aug_vect = np.concatenate((one_element_array_vect,x),axis=1)
        # tranpose to ensure that w and grad_eval are of the same shape
        g_p_vect = e_error_vect * x_p_aug_vect.T
        grad_eval = 2/P_no_of_obs*np.sum(g_p_vect,axis = 1)
        grad_norm = np.linalg.norm(grad_eval)
        # check that magnitude of gradient is not too small, if yes pick a random direction to move
        if grad_norm == 0:
            # pick random direction and normalize to have unit legnth
            grad_eval = 10**-6*np.sign(2*np.random.rand(len(w)) - 1)
            grad_norm = np.linalg.norm(grad_eval)
            grad_eval /= grad_norm
        w = w - alpha*grad_eval[:,np.newaxis]
        cost_history.append(applicable_cost_func(w, x, y))
        weight_history.append(w)
    return weight_history,cost_history