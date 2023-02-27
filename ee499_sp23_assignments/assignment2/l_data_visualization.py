import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import os   # to move around files
import sys

# custom modules
import dataset_helper as dshelp 
ip_file_4 = 'breast_cancer.npz'
data_as2 = os.path.abspath("./ee499_sp23_assignments/data/hw2_data/")
input_file_4 = os.path.join(data_as2,ip_file_4)
npzfile_4 = np.load(input_file_4)
x_train_4, x_test_4, y_train_4, y_test_4 = dshelp.load_data_npz(npzfile_4, labels_train = "labels_train", labels_test = "labels_test")

column_names = [f"Feature {x}" for x in range(1,31)]
# constructing the dataframe
df_train_4 = pd.DataFrame(data = x_train_4, columns = column_names)
df_train_4['Label'] = y_train_4
df_test_4 = pd.DataFrame(data = x_test_4, columns = column_names)
df_test_4['Label'] = y_test_4

# printing the statistics of the data
#df_train_4.info()
#df_test_4.info()
#dshelp.plot_histo_classes(df_train_4, df_test_4)
#df_train_4.describe()
#whereas in jupyter notebook the results are printed below the cells, but in .py you have to explicitly print the results 
#print(df_train_4.describe())

#extracing the labels and removing the labels columns from  the data frame  
y_df_train_4 = df_train_4.Label; y_df_test_4 = df_test_4.Label
list_drp = ["Label"]
x_df_train_4 = df_train_4.drop(list_drp,axis = 1); x_df_test_4 = df_test_4.drop(list_drp,axis = 1 )

# figurimng out the data mean and printing the same highlighting the need for the normalization
data_mean = x_df_train_4.describe().loc['mean']
data_mean.plot(kind='bar', figsize=(14,6))
plt.title('Mean Plot of Features', fontsize=16)
plt.show()

# StandardScaler() will normalize the features i.e. each feature of X, INDIVIDUALLY, so that each 
# column/feature/variable will have μ = 0 and σ = 1
# Keep in mind that all scikit-learn machine learning (ML) functions expect as input an numpy array X 
# with that shape i.e. the rows are the samples and the columns are the features/variables.
std_scaler = StandardScaler()
#x_df_train_4_normalized = x_df_train_4.apply(lambda iterator: ((iterator - iterator.mean())/iterator.std()).round(2))
x_df_train_4_normalized = pd.DataFrame(std_scaler.fit_transform(x_df_train_4), columns=x_df_train_4.columns)
data_mean = x_df_train_4_normalized.describe().loc['mean']
data_mean.plot(kind='bar', figsize=(14,6))
plt.title('Mean Plot of Normalized Features', fontsize=16)
plt.show()

# box plot
plt.rcParams["figure.figsize"] = [25, 10]
ax = x_df_train_4_normalized.boxplot(column=column_names[0:30]) 
ax.set_xticklabels(column_names, rotation=45)
plt.show()

dshelp.draw_distributions_box_plot(y_df_train_4,x_df_train_4_normalized, start_feature=0, end_feature=10)
dshelp.draw_distributions_box_plot(y_df_train_4,x_df_train_4_normalized, start_feature=10, end_feature=20 )
dshelp.draw_distributions_box_plot(y_df_train_4,x_df_train_4_normalized, start_feature=20, end_feature=31)

dshelp.plot_pairplot(y_df_test_4, x_df_test_4, start_feature=0, end_feature=10)
dshelp.plot_pairplot(y_df_test_4, x_df_test_4, start_feature=11, end_feature=20)
dshelp.plot_pairplot(y_df_test_4, x_df_test_4, start_feature=21, end_feature=30)

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x_df_train_4_normalized.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
