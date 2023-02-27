import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import os   # to move around files
import sys

def check_dimensions(x_train, x_test, y_train, y_test):
    print('Size of x_train dataset: ', x_train.shape)
    print('Size of y_train dataset: ', y_train.shape)
    print('Size of x_test dataset: ', x_test.shape)
    print('Size of y_test dataset: ', y_test.shape)

def check_nan(x_train, x_test, y_train, y_test):
    print('does x_train contain nan? ', np.isnan(np.sum(x_train)))
    print('does x_test contain nan? ', np.isnan(np.sum(x_test)))
    print('does y_train contain nan? ', np.isnan(np.sum(y_train)))
    print('does y_test contain nan? ', np.isnan(np.sum(y_test)))

def load_data_npz(npzfile, labels_train, labels_test):
    x_train = npzfile['x_train']; x_test = npzfile['x_test']; y_train = npzfile[labels_train]; y_test = npzfile[labels_test]
    check_dimensions(x_train, x_test, y_train, y_test)
    check_nan(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test

def plot_data(df_train, df_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    col_2C_train = np.where(df_train['Label']==1,'r',np.where(df_train['Label']==2,'g', 'k'))
    ax1.scatter(df_train['X1'],df_train['X2'],c= col_2C_train, s=40, edgecolors='k')
    ax1.set_title('Train Dataset',fontsize=16)
    ax1.set_xlabel('X1', fontsize = 16)
    ax1.set_ylabel('X2', fontsize = 16)
    col_2C_test = np.where(df_test['Label']==1,'r',np.where(df_test['Label']==2,'g', 'k'))
    ax2.scatter(df_test['X1'],df_test['X2'],c= col_2C_test, s=40, edgecolors='k')
    ax2.set_title('Test Dataset',fontsize=16)
    ax2.set_xlabel('X1', fontsize = 16)
    ax2.set_ylabel('X2', fontsize = 16)
    plt.show()

def plot_histo_classes(df_train, df_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    train_data_class_1 = df_train[df_train['Label'] == 1]
    train_data_class_2 = df_train[df_train['Label'] == 2]
    train_data_class_1_2 = {'Class 1': [train_data_class_1.shape[0]], 'Class 2': [train_data_class_2.shape[0]]}
    train_df_class_1_2 = pd.DataFrame(data=train_data_class_1_2)
    train_df_class_1_2.plot(kind='bar', ax=ax1, color={"Class 1": "red", "Class 2": "green"}, fontsize=16)
    ax1.set_title('Train Dataset', fontsize=16)
    ax1.set_xticks([])
    test_data_class_1 = df_test[df_test['Label'] == 1]
    test_data_class_2 = df_test[df_test['Label'] == 2]
    test_data_class_1_2 = {'Class 1': [test_data_class_1.shape[0]], 'Class 2': [test_data_class_2.shape[0]]}
    test_df_class_1_2 = pd.DataFrame(data=test_data_class_1_2)
    test_df_class_1_2.plot(kind='bar', ax=ax2, color={"Class 1": "red", "Class 2": "green"}, fontsize=16)
    ax2.set_title('Test Dataset', fontsize=16)
    ax2.set_xticks([])
    plt.show()

def create_dataframe(x_train, x_test, y_train, y_test, plot_flag = False):
    df_train = pd.DataFrame({'X1': x_train[:,0],'X2': x_train[:,1], 'Label': y_train})
    df_test = pd.DataFrame({'X1': x_test[:,0],'X2': x_test[:,1], 'Label': y_test})
    print("Total Null values count in train: ",df_train.isnull().sum().sum())
    print("Total Null values count in test: ",df_test.isnull().sum().sum())
    if (plot_flag):
        plot_data(df_train, df_test)
        plot_histo_classes(df_train, df_test)
    return df_train, df_test

def create_dataframe_bc(x_train, x_test, y_train, y_test, plot_flag = False):
    labels = []
    for i in range(1,x_train.shape[1]+1):
        labels.append("X" + str(i))
    df_train = pd.DataFrame(x_train, columns=labels)
    df_train['Label'] = y_train
    #df_train = pd.DataFrame({'X1': x_train[:,0],'X2': x_train[:,1], 'Label': y_train})
    #df_test = pd.DataFrame({'X1': x_test[:,0],'X2': x_test[:,1], 'Label': y_test})
    df_test = pd.DataFrame(x_test, columns=labels)
    df_test['Label'] = y_test
    print("Total Null values count in train: ",df_train.isnull().sum().sum())
    print("Total Null values count in test: ",df_test.isnull().sum().sum())
    if (plot_flag):
        plot_data(df_train, df_test)
        plot_histo_classes(df_train, df_test)
    return df_train, df_test


def draw_distributions_box_plot(y_df, x_df, start_feature, end_feature):
    '''
    When using hue nesting with a variable that takes two levels, setting split to True will draw half of a violin for each level. 
    This can make it easier to directly compare the distributions.
    inner="quart" will draw the quartiles of the distribution
    '''
    data = pd.concat([y_df,x_df.iloc[:,start_feature:end_feature]],axis=1)
    data = pd.melt(data,id_vars="Label",
                        var_name="Features",
                        value_name='Value')
    plt.figure(figsize=(10,10))
    sns.violinplot(x="Features", y="Value", hue="Label", data=data,split=True, inner="quart",palette ="Set2") #Draw a combination of boxplot and kernel density estimate.
    plt.xticks(rotation=90)
    plt.show()

def plot_pairplot(y_df, x_df, start_feature, end_feature):
    data = pd.concat([y_df,x_df.iloc[:,start_feature:end_feature]],axis=1)
    grid=sns.pairplot(data=data,kind ="scatter",hue="Label",palette="Set1")
    plt.show()

def plot_contours(g, weight_history, title, data, labels, inputs_flag = False):
    weights_steps_x = np.array([i[0] for i in weight_history])
    weights_steps_y = np.array([i[1] for i in weight_history])
    x = y = np.arange(-4.5, 4.5, 0.05)
    X, Y = np.meshgrid(x, y)
    if (inputs_flag):
        zs = np.array([g(np.array([x,y]), data, labels) for x,y in zip(np.ravel(X), np.ravel(Y))])
    else:
        zs = np.array([g(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    fig = plt.figure(figsize=(5,5))
    cp = plt.contourf(X, Y, Z, cmap='coolwarm')
    plt.colorbar(cp)
    plt.xlabel(r'$w_1$', fontsize=12)
    plt.ylabel(r'$w_2$', fontsize=12)
    #plt.axhline(y=0, color='r', linestyle='--')
    #plt.axvline(x=0, color='r', linestyle='--')
    plt.plot(weights_steps_x, weights_steps_y, 'sk', markersize=4)
    plt.quiver(weights_steps_x[:-1], weights_steps_y[:-1], weights_steps_x[1:]-weights_steps_x[:-1], weights_steps_y[1:]-weights_steps_y[:-1], scale_units='xy', angles='xy', scale=1)
    plt.plot(weights_steps_x[-1], weights_steps_y[-1], 'sy')
    plt.title(title,fontsize=16)
    plt.show()