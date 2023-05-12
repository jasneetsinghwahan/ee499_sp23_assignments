import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os       # navigate dir. structure
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime       # for naming the graphs with timestamps
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback 
import copy
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

#-------------------------- 
# hyper-parameters for our model
#-------------------------- 
train_per = 75
BATCH_SIZE = 64
EPOCH_DEFAULT = 500
num_folds = 3          # Define the number of splits 
reg_val = 0.001         # regularizer
fold_losses = []
fold_misclassifications = []
models = []
auc_rocs = []
balanced_accuracies = []
balanced_precisions = []
balanced_recalls = []
weighted_losses = []
best_model_index = None
best_val_loss = float('inf')

#-------------------------- 
# parse the command line 
#-------------------------- 
cur_time = datetime.now()
time_string = cur_time.strftime("%m-%d_%H:%M:%S")
fn_time_string = time_string.replace(':', '_').replace('-', '_')
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_name', default = 'default', help='all of the output files will be created based on this name')
parser.add_argument('-e', '--epochs', type=int, default=EPOCH_DEFAULT, help='number of epochs to run [default=1]')
parser.add_argument('-p', '--plot', action='store_true', default=True, help='plot training history [default=true]')
args = parser.parse_args()
run_name = args.run_name
model_folder = "artresults"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
WEIGHT_FILE = './artresults/' + run_name  + fn_time_string + '_weights.h5'
MODEL_FILE = './artresults/' + run_name + fn_time_string + '_model.h5'
LEARNCURVE_FILE = './artresults/' + run_name + fn_time_string + '_learning_curve.jpg'
CONFUSMAT_FILE = './artresults/' + run_name + fn_time_string + '_confusion_matrix.jpg'
MODEL_SHAPE = './artmodel/' + run_name + fn_time_string + '_shape.png'
LOG_FILE = './artresults/' + run_name + fn_time_string + '.csv'
PRUNINGACC_FILE = './artresults/'+ run_name + fn_time_string + '_prunedaccuracy.png'
DO_PLOT = args.plot
EPOCHS = args.epochs

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#-------------------------- 
# load the dataset
#-------------------------- 
artip_data_file = './artrawdata/op_artrawdata.csv'
df_artdata = pd.read_csv(artip_data_file)
data = df_artdata.iloc[:,:-1]       # All rows, and all columns except the last one
labels = df_artdata.iloc[:,-1]      # All rows, and only the last column

#-------------------------- 
# model architecture
#-------------------------- 
def modeltf(input_dim):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val), input_dim=input_dim))
    model.add(Dense(12, activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val)))
    model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val)))
    model.add(Dense(1, activation ='sigmoid'))
    opt = Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'], weighted_metrics=[BinaryCrossentropy()])
    print("Model summary:")
    model.summary()
    return model

#-----------------
# callbacks
#------------------
class MisclassificationCallback(Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.train_misclassifications = []
        self.val_misclassifications = []
        
    def on_epoch_end(self, epoch, logs=None):
        y_train_preds = (self.model.predict(self.X_train) > 0.5).astype(int).flatten()
        train_misclassification_count = np.sum(y_train_preds != self.y_train)
        self.train_misclassifications.append(train_misclassification_count)

        y_val_preds = (self.model.predict(self.X_val) > 0.5).astype(int).flatten()
        val_misclassification_count = np.sum(y_val_preds != self.y_val)
        self.val_misclassifications.append(val_misclassification_count)

#-------------------------- 
# training and validation using stratrified kfold
#-------------------------- 
model = modeltf(df_artdata.shape[1] - 1)

cv = StratifiedKFold(n_splits=num_folds)

# split the data into training and testing
train_ratio = train_per / 100
temp_ratio = 1 - train_ratio
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=temp_ratio, random_state=42)

early_stopping_epochs = []

fold_count = 0
for fold, (train_index, val_index) in enumerate(cv.split(train_data, train_labels)):
    X_train, X_val = train_data.iloc[train_index], train_data.iloc[val_index]
    y_train, y_val = train_labels.iloc[train_index], train_labels.iloc[val_index]

    # Compute the class weights for the current fold
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    class_weights = (1 / class_counts) * len(y_train) / len(unique_classes)
    class_weights_dict = dict(zip(unique_classes, class_weights))
    total_samples = class_counts[0] + class_counts[1]
    percentage_distribution = class_counts / total_samples * 100
    print(f"for fold {fold_count+1} % distribution for class {unique_classes[0]} {percentage_distribution[unique_classes[0]]:.2f} and class {unique_classes[1]} {percentage_distribution[unique_classes[1]]:.2f}")
    model = modeltf(train_data.shape[1])
    misclassification_callback = MisclassificationCallback(X_train, y_train, X_val, y_val)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    H = model.fit(
        X_train, y_train, 
        batch_size=BATCH_SIZE, 
        validation_data=(X_val, y_val), 
        epochs=EPOCHS, 
        class_weight=class_weights_dict,
        callbacks=[misclassification_callback, early_stopping_callback],
        verbose=1)
    
    # Save the current model
    models.append(copy.deepcopy(model))

    fold_losses.append(H.history)
    fold_misclassifications.append((misclassification_callback.train_misclassifications, misclassification_callback.val_misclassifications))

    # Calculate the weighted loss
    sample_weights = np.array([class_weights_dict[label] for label in y_val])
    weighted_loss = model.evaluate(X_val, y_val, sample_weight=sample_weights, verbose=0)[0]

    # Update the best model index and best validation loss
    if weighted_loss < best_val_loss:
        best_val_loss = weighted_loss
        best_model_index = fold

    if early_stopping_callback.stopped_epoch == EPOCHS - 1:
        print(f"Fold {fold + 1}: All epochs completed.")
    else:
        print(f"Fold {fold + 1}: Early stopping occurred at epoch {early_stopping_callback.stopped_epoch + 1}.")
        early_stopping_epochs.append(early_stopping_callback.stopped_epoch + 1)


# Load the best model
best_model = models[best_model_index]
print(f"Best model is from Fold {best_model_index + 1} with validation loss {best_val_loss:.2f}")

plt.figure(figsize=(24, 10))

for fold in range(num_folds):
    num_epochs = len(fold_losses[fold]['loss'])
    epochs = range(1, num_epochs + 1)
    plt.subplot(2, num_folds, fold + 1)
    plt.plot(epochs, fold_losses[fold]['loss'], label='Training Loss')
    plt.plot(epochs, fold_losses[fold]['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.title(f'Fold {fold + 1} Loss')
    plt.title(f'Fold {fold + 1} Loss (Early stopping at epoch {early_stopping_epochs[fold]})')
    plt.legend()

    min_val_loss = min(fold_losses[fold]['val_loss'])
    min_val_loss_idx = fold_losses[fold]['val_loss'].index(min_val_loss)
    plt.annotate(f'Min Val Loss: {min_val_loss:.2f}', xy=(min_val_loss_idx + 1, min_val_loss), xytext=(min_val_loss_idx - 5, min_val_loss + 0.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.subplot(2, num_folds, num_folds + fold + 1)
    plt.plot(epochs, fold_misclassifications[fold][0], label='Training Misclassifications')
    plt.plot(epochs, fold_misclassifications[fold][1], label='Validation Misclassifications')
    plt.xlabel('Epochs')
    plt.ylabel('Misclassifications')
    #plt.title(f'Fold {fold + 1} Misclassifications')
    plt.title(f'Fold {fold + 1} Misclassifications (Early stopping at epoch {early_stopping_epochs[fold]})')
    plt.legend()

    min_val_misclassification = min(fold_misclassifications[fold][1])
    min_val_misclassification_idx = fold_misclassifications[fold][1].index(min_val_misclassification)
    plt.annotate(f'Min Val Misclassification: {min_val_misclassification}', xy=(min_val_misclassification_idx + 1, min_val_misclassification),
                 xytext=(min_val_misclassification_idx - 5, min_val_misclassification + 5), arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.tight_layout()
plt.savefig(LEARNCURVE_FILE)
plt.show()

# Evaluate the best model on the test set
test_metrics = best_model.evaluate(test_data, test_labels, verbose=0)
test_loss, test_accuracy = test_metrics[0], test_metrics[1]
print(f'Test Loss: {test_loss:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

# Make predictions on the test set
test_pred_labels = (best_model.predict(test_data) > 0.5).astype(int).flatten()

# Calculate the number of misclassifications
num_misclassifications = np.sum(test_labels != test_pred_labels)
print(f'Number of misclassifications: {num_misclassifications}')

# Calculate the confusion matrix
cm = confusion_matrix(test_labels, test_pred_labels)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar=False,
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.savefig(CONFUSMAT_FILE)
plt.show()

import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
import heapq
import copy
from keras.models import clone_model
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import clone_model

def prune_weights(weights, sparsity):
    flattened_weights = np.abs(weights.flatten())
    threshold = np.percentile(flattened_weights, sparsity * 100)
    pruned_weights = np.where(np.abs(weights) < threshold, 0, weights)
    return pruned_weights

def prune_model(model, sparsity):
    pruned_model = clone_model(model)
    pruned_model.set_weights(model.get_weights())

    for i, layer in enumerate(pruned_model.layers[:-1]):
        layer_weights = layer.get_weights()
        if len(layer_weights) > 0:
            pruned_weights = prune_weights(layer_weights[0], sparsity)
            layer.set_weights([pruned_weights, layer_weights[1]])

    return pruned_model

pruning_levels = np.linspace(0, 0.9, 2)
pruning_results = []

for level in pruning_levels:
    pruned_model = prune_model(best_model, level)
    pruned_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    test_loss, test_accuracy = pruned_model.evaluate(test_data, test_labels, verbose=0)
    test_preds = pruned_model.predict(test_data)
    test_pred_labels = (test_preds > 0.5).astype(int).flatten()
    misclassifications = np.sum(test_pred_labels != test_labels)

    pruning_results.append({
        'Prune Percentage': level,
        'Loss': test_loss,
        'Accuracy': test_accuracy,
        'Misclassifications': misclassifications
    })


# Extract results for plotting
num_params = [573 - int(573 * prune_percentage) for prune_percentage in np.linspace(0, (573 - 50) / 573, 2)]
losses = [result['Loss'] for result in pruning_results]
accuracies = [result['Accuracy'] for result in pruning_results]
misclassifications = [result['Misclassifications'] for result in pruning_results]

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# 1st subplot - Loss
ax1.plot(num_params, losses, label='Loss', marker='o')
ax1.set_xlabel('Number of Parameters')
ax1.set_ylabel('Loss')
ax1.set_title('Loss vs Pruned Model')

# 2nd subplot - Accuracy
ax2.plot(num_params, accuracies, label='Accuracy', marker='o', color='orange')
ax2.set_xlabel('Number of Parameters')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy vs Pruned Model')

# 3rd subplot - Misclassifications
ax3.plot(num_params, misclassifications, label='Misclassifications', marker='o', color='purple')
ax3.set_xlabel('Number of Parameters / % Sparsity')
ax3.set_ylabel('Misclassifications')
ax3.set_title('Misclassifications vs Pruned Model')
plt.savefig(PRUNINGACC_FILE)
plt.show()