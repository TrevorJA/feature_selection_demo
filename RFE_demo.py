# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:42:18 2023

@author: tja73
"""

from NeuralNetRFE import NeuralNetRFE
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from NeuralNetRFE import NeuralNetRFE
import matplotlib.pyplot as plt

# Load training input data, output data, and feature names
x_train = np.loadtxt(f'./data/standardized_training_inputs.csv', delimiter = ',')
y_train = np.log(np.loadtxt(f'./data/training_output.csv', delimiter= ','))
feature_names = np.loadtxt('./data/training_data_names.csv', delimiter = ',', dtype='str')
y_train = np.exp(y_train)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Initialize the NeuralNetRFE object
nn_rfe = NeuralNetRFE(n_features=x_train.shape[1], hidden_layer_sizes=(100, 50), importance='weights', max_nn_iter = 1000)


    
# Train a neural network with the selected features

n_repeats = 5
selected_feature_indices_across_test = []
xTr = x_train.copy()
xTe = x_test.copy()
repeat_mse = np.zeros(n_repeats)
repeat_nmse = np.zeros(n_repeats)

for r in range(n_repeats):
    print(f'Running repeat number {r}')
    nn_rfe = NeuralNetRFE(n_features=xTr.shape[1], hidden_layer_sizes=(100, 50), importance='weights', max_nn_iter = 200)
    selected_features = nn_rfe.feature_selection(xTr, y_train, n_features_to_select=34)
    selected_feature_indices = nn_rfe.selected_feature_indices
    
    nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter = 200)
    nn.fit(selected_features, y_train)

    # Evaluate the model on the test set
    x_test_selected = x_test[:, selected_feature_indices]
    y_pred = nn.predict(x_test_selected)
    repeat_mse[r] = ((y_test - y_pred)**2).mean()

    selected_feature_indices_across_test.append(selected_feature_indices)   
    
selected = np.hstack([selected_feature_indices_across_test[i] for i in range(len(selected_feature_indices_across_test))])
unique, counts = np.unique(selected, return_counts=True)

final_features = unique[counts>=3]

x_selected= x_train[:,final_features]
x_test_selected = x_test[:, final_features]

nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter =200)
nn.fit(x_selected, y_train)

# Evaluate the model on the test set
y_pred = nn.predict(x_test_selected)
mse = ((y_test - y_pred)**2).mean()

plt.scatter(np.log(y_pred), np.log(y_test))


plt.scatter(np.exp(y_pred), np.exp(y_test))




# Find the preferred number of features
check_n_features = np.flip(np.arange(1,x_train.shape[1]))
max_iter = 200
test_mse = []
test_mpe = []

xTr = x_train.copy()
xTe = x_test.copy()

for n in check_n_features:
    print(f'Running selection for {n} features.')
    nn_rfe = NeuralNetRFE(n_features=xTr.shape[1], hidden_layer_sizes=(100, 50), importance='weights', max_nn_iter = max_iter)
    selected_features = nn_rfe.feature_selection(xTr, y_train, n_features_to_select=n)
    selected_feature_indices = nn_rfe.selected_feature_indices
    
    nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter = max_iter)
    nn.fit(selected_features, y_train)
    x_test_selected = xTe[:, selected_feature_indices]
    y_pred = nn.predict(x_test_selected)
    test_mpe.append(np.abs((y_test - y_pred) / y_pred).mean())
    test_mse.append(((y_test - y_pred)**2).mean())
    xTr = selected_features
    xTe = x_test_selected
    
plt.scatter(check_n_features,test_mse)
plt.ylim([0,500])





"""
# Select the top 5 features using Recursive Feature Elimination
selected_features = nn_rfe.feature_selection(x_train, y_train, n_features_to_select=100)
selected_feature_indices = nn_rfe.selected_feature_indices


nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter =100)
nn.fit(x_train, y_train)

# Evaluate the model on the test set
y_pred = nn.predict(x_test)
mse = ((y_test - y_pred)**2).mean()
nmse = ((np.exp(y_test) - np.exp(y_pred))**2).mean()
plt.scatter(y_pred, y_test)
plt.scatter(np.exp(y_pred), np.exp(y_test))


# Train a neural network with the selected features
iter_mse = []
iter_nmse = []
n_repeats = 10

repeat_mse = np.zeros(n_repeats)
repeat_nmse = np.zeros(n_repeats)
for r in range(n_repeats):
    nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter = 200)
    nn.fit(selected_features, y_train)

    # Evaluate the model on the test set
    x_test_selected = x_test[:, selected_feature_indices]
    y_pred = nn.predict(x_test_selected)
    repeat_mse[r] = ((y_test - y_pred)**2).mean()
    #repeat_nmse[r] = ((np.exp(y_test) - np.exp(y_pred))**2).mean()
#iter_nmse.append(repeat_nmse.mean())
iter_mse.append(repeat_mse.mean())
    
# Check the performance across a range of feature numbers
check_n_features = [130, 120, 110, 100, 90, 80, 70, 60, 50]
selected_feature_indices_across_test = []
max_iter = 500
test_mse = []
test_mpe = []

xTr = x_train.copy()
xTe = x_test.copy()

for n in check_n_features:
    print(f'Running selection for {n} features.')
    nn_rfe = NeuralNetRFE(n_features=xTr.shape[1], hidden_layer_sizes=(100, 50), importance='weights', max_nn_iter = max_iter)
    selected_features = nn_rfe.feature_selection(xTr, y_train, n_features_to_select=n)
    selected_feature_indices = nn_rfe.selected_feature_indices
    
    nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter = max_iter)
    nn.fit(selected_features, y_train)
    x_test_selected = xTe[:, selected_feature_indices]
    y_pred = nn.predict(x_test_selected)
    test_mpe.append(np.abs((y_test - y_pred) / y_pred).mean())
    test_mse.append(((y_test - y_pred)**2).mean())
    selected_feature_indices_across_test.append(selected_feature_indices)

selected = np.hstack([selected_feature_indices_across_test[i] for i in range(len(selected_feature_indices_across_test))])

"""