import os
current_dir = os.getcwd()

# Specify the path to go to from default directory
sub_dir = "ee499_sp23_assignments/learn_by_doing"
new_dir = os.path.join(current_dir, sub_dir)
os.chdir(new_dir)
readDataPath = '../../ee499_ml_spring23/readData/'

# code to check if the path exists
if os.path.exists(readDataPath):
    print(f"{readDataPath} exists.")

#files = os.listdir(new_dir)
#
#for file in files:
#    print(file)


# Print the directories
#print(directories)
#data = np.loadtxt(readDataPath + '2d_classification_data_v1_entropy.csv',delimiter = ',')