import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras.optimizers import Adam

import pandas as pd
import numpy as np
import os       # navigate dir. structure


os.chdir(os.path.dirname(os.path.abspath(__file__)))
pretrainweight_file = './preweights.csv'
artweights_proc_file = './artweights.h5'
artip_data_file = './artrawdata/op_artrawdata.csv'

#---------------------
# store the weights from csv file to .h5 file
#---------------------
# Read the CSV file
df = pd.read_csv(pretrainweight_file, header=None)

# Extract the weights and biases
weights_1 = df.iloc[:15, :10].values
biases_1 = df.iloc[15, :10].values
weights_2 = df.iloc[16, :10].values.reshape(10, 1)
biases_2 = df.iloc[17, :1].values

# Recreate the neural network architecture
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=input_dim))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model with a loss function, optimizer, and metrics
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

input_dim = 15  # Set this to the number of input features
model = create_model(input_dim)

# Summary of the model's architecture
print("Model summary:")
model.summary()

# Get the model's layers
print("\nModel layers:")
for i, layer in enumerate(model.layers):
    print(f"Layer {i+1}: {layer}")

#Set the weights for each layer
model.layers[0].set_weights([weights_1, biases_1])
model.layers[1].set_weights([weights_2, biases_2])

#Save the model with the loaded weights to an HDF5 file
model.save_weights(artweights_proc_file)

#---------------------
# verify that weights are loaded correctly
#---------------------
layer_1_weights = model.layers[0].get_weights()[0]
layer_1_biases = model.layers[0].get_weights()[1]
layer_2_weights = model.layers[1].get_weights()[0]
layer_2_biases = model.layers[1].get_weights()[1]

# Compare each element of the arrays
l1_w_equal =  weights_1.astype(np.float32) == layer_1_weights
l1_b_equal =  biases_1.astype(np.float32) == layer_1_biases
l2_w_equal =  weights_2.astype(np.float32) == layer_2_weights
l2_b_equal =  biases_2.astype(np.float32) == layer_2_biases

# Count the number of unequal elements
l1_w_num_unequal = np.sum(~l1_w_equal)
l1_b_num_unequal = np.sum(~l1_b_equal)
l2_w_num_unequal = np.sum(~l2_w_equal)
l2_b_num_unequal = np.sum(~l2_b_equal)

print(f'layer 1 weights have {l1_w_num_unequal} unequal elements.')
print(f'layer 1 biases have {l1_b_num_unequal} unequal elements.')
print(f'layer 2 weights have {l2_w_num_unequal} unequal elements.')
print(f'layer 2 biases have {l2_b_num_unequal} unequal elements.')

#---------------------
# generate the y_hat from the trained data
#---------------------
# Read the artificially generated data from the CSV file
pd_artdata = pd.read_csv(artip_data_file)
# Get the number of rows and columns in the DataFrame
dataset_size, num_cols = pd_artdata.shape

#Use the model to predict the output (y_hat)
X = pd_artdata.values

# Predict the output (y_hat)
y_hat = model.predict(X)

# Convert the predictions to binary labels (0 or 1)
col_name = 'can_migrate'
y_hat_binary = (y_hat > 0.5).astype(int)

def save_to_csv(data, column_name, file_name):
    df = pd.DataFrame(data, columns=[column_name])
    try:
        existing_df = pd.read_csv(file_name)
        combined_df = pd.concat([existing_df, df], axis=1)
    except FileNotFoundError:
        combined_df = df
    combined_df.to_csv(file_name, index=False)

save_to_csv(y_hat_binary, col_name, artip_data_file)
print(f"{dataset_size} elements for '{col_name}' written to column number {num_cols+1}")