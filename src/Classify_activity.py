

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from Evaluate_classifier import load_sensor_data

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_suffix = '_train_1.csv'
test_suffix = '_train_2.csv'
train_end_index = 3511

# Logistic regression hyperparameters
C = 1
l1_ratio = 0.9
max_iter = int(1e4)

def predict_test(train_data, train_labels, test_data):
    # Feature extraction: compute mean and standard deviation of each row for
    # each sensor and concatenate across sensors to form the feature vector
    mean_train_feature = np.mean(train_data, axis=1)
    std_train_feature = np.std(train_data, axis=1)
    train_features = np.hstack((mean_train_feature, std_train_feature))
    mean_test_feature = np.mean(test_data, axis=1)
    std_test_feature = np.std(test_data, axis=1)
    test_features = np.hstack((mean_test_feature, std_test_feature))

    # Standardize features and train a logistic regression model
    scaler = StandardScaler()
    train_features_std = scaler.fit_transform(train_features)
    test_features_std = scaler.transform(test_features)
    
    return test_outputs

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    # Load labels and sensor data into 3-D array
    train_labels = np.loadtxt('labels' + train_suffix, dtype='int')
    train_data = load_sensor_data(sensor_names, train_suffix)
    test_labels = np.loadtxt('labels' + test_suffix, dtype='int')
    test_data = load_sensor_data(sensor_names, test_suffix)

    # Predict activities on test data
    test_outputs = predict_test(train_data, train_labels, test_data)

    # Compute micro and macro-averaged F1 scores
    micro_f1 = f1_score(test_labels, test_outputs, average='micro')
    macro_f1 = f1_score(test_labels, test_outputs, average='macro')
    print(f'Micro-averaged F1 score: {micro_f1}')
    print(f'Macro-averaged F1 score: {macro_f1}')

    # Examine outputs compared to labels
    n_test = test_labels.size
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(n_test), test_labels, 'b.')
    plt.xlabel('Time window')
    plt.ylabel('Target')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(n_test), test_outputs, 'r.')
    plt.xlabel('Time window')
    plt.ylabel('Output (predicted target)')
    plt.show()
    