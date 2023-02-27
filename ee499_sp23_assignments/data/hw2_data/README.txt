Notes on data:

All but the housing data is stored in .npz files.  These are compressed data archives.  You can load thes via:

data = np.load('foo.npz')

To see the names of the numpy arrays stored in the file:

list(data.keys())

For the classification datasets, you will see keys:

x_train, labels_train, x_test, labels_test

which have obvious meeaning.

For the regression dataset, you will see:

x_train, y_train, x_test, y_test

which again should have obvious meaning.  


BREAST CANCER DATA:
---------------------

For the breast cancer data, each data point is 30 dimensions.  The label 1 is for Malignent and label 2 is for benign.  To learn more about this dataset, see: 

https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29

Specifically, the data we are using is the first wdbc.data file here, which is described in the first wdbc.names file.  Both of these are text files if you wish to explore.  The .data is a csv and the .names is just a test description.  Rename the .data as .csv and the .names as .txt to open with your standard applications.  We have cleaned up this data for you a bit and that's what's in the .npz file.


CA HOUSING DATA:
-----------------

This is a .csv.  You can open in excel to explore.  It is 20640 rows of data.  The top row describes the data fields.  Your task is to use the data available to predict the housing price.  Note that the ocean_proximity field is text.  You will need to do some cleaning on this.  Specifically, determine how many different values for this are in the data and then assign a numerical value to each.  The exact numerical value for each is up to you.

You can read the .csv file into Python using the csv package (import csv) or you may like the pandas package, which is very powerful.  

Alexios has prepared a Python notebook for you that overviews the most common data formats and tools for data.  He can help in you with this in discussion.  

Note: we purposely did not "clean" the housing data to give you a sense of this task -- always a part of real-world ML design!

SHUFFLE THE DATA:  once you have the data cleaned, you will have a numpy array like:

x with shape (20640, 9)
y with shape (20640, 1)  -- i.e., the price

Use the permuation in housing_permutation.npz to shuffle the data:

x = x[perm]
y = y[perm]

Then, split the data into train and test with the first 17540 being train and the last 3100 being test.  Make sure you shuffle both the x and y!