# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:42:18 2023

@author: tja73
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm
# Custom modules
from NeuralNetRFE import NeuralNetRFE
from plot_scripts import plot_distribution, plot_multiple_distributions, plot_predicted_vs_actual

rerun_all = False

# Load training input data, output data, and feature names
x = np.loadtxt(f'./data/standardized_training_inputs.csv', delimiter = ',')
y = np.loadtxt(f'./data/training_output.csv', delimiter= ',')
feature_names = np.loadtxt('./data/training_data_names.csv', delimiter = ',', dtype='str')

dam_indices = np.argwhere(feature_names == 'CAT_NID_STORAGE2013')

x[:,129] 


# Visualze data dsitributions
if rerun_all:
    plot_distribution(y, save = True)
    plot_distribution(np.log(y), log = True, save = True)
    plot_multiple_distributions(x, 5, feature_names)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

### Baseline model performance (with all features)
# Initialize and train MLP
nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter = 1000)
nn.fit(x_train, y_train)

# Make prediction and get MSE
y_pred = nn.predict(x_test)
baseline_mse = ((y_test - y_pred)**2).mean()

# Visualize predictions
plot_predicted_vs_actual(np.log(y_test), np.log(y_pred))
plot_predicted_vs_actual(y_test, y_pred)

# Repeat for log
nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter = 1000)
nn.fit(x_train, np.log(y_train))
y_pred = nn.predict(x_test)
mse_log = ((y_test - np.exp(y_pred))**2).mean()
plot_predicted_vs_actual(np.log(y_test), y_pred)
plot_predicted_vs_actual(y_test, np.exp(y_pred))


# Find the preferred number of features
check_n_features = np.flip(np.arange(1,x_train.shape[1]))
max_iter = 200
test_log_errors = np.zeros((y_test.shape[0], len(check_n_features)))
xTr = x_train.copy()
xTe = x_test.copy()

for i,n in enumerate(check_n_features):
    print(f'Running selection for {n} features.')
    nn_rfe = NeuralNetRFE(hidden_layer_sizes=(100, 50))
    selected_features = nn_rfe.feature_selection(xTr, np.log(y_train), n_features_to_select=n)
    selected_feature_indices = nn_rfe.selected_feature_indices

    nn = MLPRegressor(hidden_layer_sizes=(100, 50))
    nn.fit(selected_features, np.log(y_train))
    x_test_selected = xTe[:, selected_feature_indices]
    y_pred = nn.predict(x_test_selected)
    test_log_errors[:,i] = (y_test - np.exp(y_pred))**2
    xTr = selected_features
    xTe = x_test_selected


std_log = np.log(np.std(test_log_errors, axis=0))
mean_log = np.log(np.mean(test_log_errors, axis =0))
q75_log = np.quantile(test_log_errors, 0.75, axis=0)
q25_log = np.quantile(test_log_errors, 0.25, axis=0)
median_log = np.median(test_log_errors, axis = 0)

plt.scatter(check_n_features, test_log_errors.mean(axis=0))
plt.ylim([0,250])

plt.scatter(check_n_features, np.log(test_log_errors.mean(axis=0)))

plt.bar(check_n_features, np.std(test_log_errors, axis=0))
#plt.ylim([0,10000])
plt.yscale('log')


# Find the preferred number of features
check_n_features = [10, 20, 40, 60, 80, 100, 120]
errors = np.zeros((y_test.shape[0], len(check_n_features)))
all_selected_features = []
xTr = x_train.copy()
xTe = x_test.copy()

for i,n in enumerate(check_n_features):
    print(f'Running selection for {n} features.')
    nn_rfe = NeuralNetRFE(hidden_layer_sizes=(100, 50))
    selected_features = nn_rfe.feature_selection(xTr, np.log(y_train), n_features_to_select=n)
    selected_feature_indices = nn_rfe.selected_feature_indices

    nn = MLPRegressor(hidden_layer_sizes=(100, 50))
    nn.fit(selected_features, np.log(y_train))
    x_test_selected = xTe[:, selected_feature_indices]
    y_pred = nn.predict(x_test_selected)
    errors[:,i] = (y_test - np.exp(y_pred))**2

    all_selected_features.append(selected_feature_indices)


plt.boxplot(errors)
plt.ylim([0,5])

plt.plot(check_n_features, np.log(median_log), color = 'darkblue')
plt.fill_between(check_n_features, np.log(q75_log), np.log(q25_log), color = 'cornflowerblue', alpha = 0.5)
plt.show()

plt.plot(check_n_features, std_log, color = 'darkblue')
plt.ylim([0,3000])



"""

# REPEAT USING LOG OUTPUT
n_repeats = 3
check_n_features = [25, 50, 75, 100, 125]
selected_feature_indices_across_test = []
xTr = x_train.copy()
xTe = x_test.copy()
errors = np.zeros((len(check_n_features), y_test.shape[0], n_repeats))

for i,n in enumerate(check_n_features):    
    for r in range(n_repeats):
        print(f'Running repeat number {r}')
        nn_rfe = NeuralNetRFE(hidden_layer_sizes=(100, 50))
        selected_features = nn_rfe.feature_selection(xTr, y_train, n_features_to_select=n)
        selected_feature_indices = nn_rfe.selected_feature_indices
    
        nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter = 200)
        nn.fit(selected_features, y_train)
    
        # Evaluate the model on the test set
        x_test_selected = x_test[:, selected_feature_indices]
        y_pred = nn.predict(x_test_selected)
        errors[i, :, r] = ((y_test - y_pred)**2)
    
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






# Initialize the NeuralNetRFE object
nn_rfe = NeuralNetRFE(n_features=x_train.shape[1], hidden_layer_sizes=(100, 50), importance='weights', max_nn_iter = 1000)



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


std_normal = np.log(np.std(test_normal_errors, axis=0))
mean_normal = np.log(np.mean(test_normal_errors, axis=0))
q75_normal = np.quantile(np.log(test_normal_errors), 0.975, axis=0)
q25_normal = np.quantile(np.log(test_normal_errors), 0.025, axis=0)
median_normal = np.median(np.log(test_normal_errors), axis = 0)

plt.scatter(check_n_features, mean_normal, color = 'darkblue')
plt.fill_between(check_n_features, mean_normal+2*std_normal, mean_normal-2*std_normal, color = 'cornflowerblue', alpha = 0.5)

plt.show()



plt.plot(check_n_features, mean_normal, color = 'darkblue')
plt.plot(check_n_features, median_normal, color = 'darkgreen')
plt.fill_between(check_n_features, q75_normal, q25_normal, color = 'cornflowerblue', alpha = 0.5)
plt.show()

check_n_features = np.flip(np.arange(1,x_train.shape[1]))
max_iter = 200
test_normal_errors = np.zeros((y_test.shape[0], len(check_n_features)))
xTr = x_train.copy()
xTe = x_test.copy()

for i,n in enumerate(check_n_features):
    print(f'Running selection for {n} features.')
    nn_rfe = NeuralNetRFE(hidden_layer_sizes=(100, 50))
    selected_features = nn_rfe.feature_selection(xTr, y_train, n_features_to_select=n)
    selected_feature_indices = nn_rfe.selected_feature_indices

    nn = MLPRegressor(hidden_layer_sizes=(100, 50))
    nn.fit(selected_features, y_train)
    x_test_selected = xTe[:, selected_feature_indices]
    y_pred = nn.predict(x_test_selected)
   
    test_normal_errors[:,i] = (y_test -y_pred)**2
    xTr = selected_features
    xTe = x_test_selected
plt.plot(check_n_features, np.log(mean_normal), color = 'darkblue')
plt.plot(check_n_features, np.log(mean_log), color = 'darkgreen')
plt.ylim([0,30])


"""
