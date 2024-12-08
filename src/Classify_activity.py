

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_suffix = '_train_1.csv'
test_suffix = '_train_2.csv'

def load_sensor_data(sensor_names, suffix):
    data_slice_0 = np.loadtxt(sensor_names[0] + suffix, delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1], len(sensor_names)))
    data[:, :, 0] = data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(sensor_names[sensor_index] + suffix, delimiter=',')

    return data

# data is loaded in the order: array containing all rows, inside a row is all the columns,
# & inside all the columns is all the different features
def predict_test(train_data, train_labels, test_data):
    # Standardize the first three Acceleration features & then standardize the last three Gyr features
    features_first_3 = train_data[:, :, :3]
    features_last_3 = train_data[:, :, 3:]
    scaler_first_3 = StandardScaler()
    scaler_last_3 = MinMaxScaler()

    for i in range(3):
        features_first_3[:, :, i] = scaler_first_3.fit_transform(features_first_3[:, :, i])

    for i in range(3):
        features_last_3[:, :, i] = scaler_last_3.fit_transform(features_last_3[:, :, i])

    standardized_data = np.concatenate([features_first_3, features_last_3], axis=2)

    # dimensions for the array should be: 5211, 60, & 6

    test_outputs = 0
    
    return test_outputs

if __name__ == "__main__":
    train_labels = np.loadtxt('labels' + train_suffix, dtype='int')
    train_data = load_sensor_data(sensor_names, train_suffix)
    test_labels = np.loadtxt('labels' + test_suffix, dtype='int')
    test_data = load_sensor_data(sensor_names, test_suffix)

    test_outputs = predict_test(train_data, train_labels, test_data)

    # micro_f1 = f1_score(test_labels, test_outputs, average='micro')
    # macro_f1 = f1_score(test_labels, test_outputs, average='macro')
    # print(f'Micro-averaged F1 score: {micro_f1}')
    # print(f'Macro-averaged F1 score: {macro_f1}')
    #
    # n_test = test_labels.size
    # plt.subplot(2, 1, 1)
    # plt.plot(np.arange(n_test), test_labels, 'b.')
    # plt.xlabel('Time window')
    # plt.ylabel('Target')
    # plt.subplot(2, 1, 2)
    # plt.plot(np.arange(n_test), test_outputs, 'r.')
    # plt.xlabel('Time window')
    # plt.ylabel('Output (predicted target)')
    # plt.show()