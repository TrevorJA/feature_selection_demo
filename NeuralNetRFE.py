# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:41:36 2023

@author: tja73
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

class NeuralNetRFE:
    def __init__(self, n_features, 
                 hidden_layer_sizes=(100, 100), 
                 importance='weights', 
                 max_nn_iter = 300):
        """
        Initialize the neural network model with specified number of features, hidden layer sizes,
        and importance scoring method.

        Parameters:
        - n_features: int, the number of features in the input data
        - hidden_layer_sizes: tuple, the size of the hidden layers in the neural network
        - importance: str, the importance scoring method to use ('weights', 'loss', 'activation')
        - n_jobs: int, the number of jobs to run in parallel when fitting the neural network
        """
        self.NN_MAX_ITER = max_nn_iter
        self.nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter = self.NN_MAX_ITER)
        self.n_features = n_features
        self.importance = importance


    def _get_activations(self, X_train):
        hidden_layer_sizes = self.nn.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        layer_units = [X_train.shape[1]] + hidden_layer_sizes + \
            [self.nn.n_outputs_]
        activations = [X_train]
        for i in range(self.nn.n_layers_ - 1):
            activations.append(np.empty((X_train.shape[0],
                                         layer_units[i + 1])))
        self.nn._forward_pass(activations)
        return activations

    def _get_importance_scores(self, X_train, y_train):
        """
        Compute the importance scores for each feature in the input data.

        Parameters:
        - X_train: array-like of shape (n_samples, n_features), the input data for training
        - y_train: array-like of shape (n_samples,), the target variable for training

        Returns:
        - scores: array-like of shape (n_features,), the importance scores for each feature
        """
        if self.importance == 'weights':
            self.train(X_train, y_train)
            scores = np.abs(self.nn.coefs_[0]).sum(axis=1)
        elif self.importance == 'loss':
            baseline_loss = self.nn.fit(X_train, y_train).loss_
            scores = []
            for i in range(self.n_features):
                X_reduced = np.delete(X_train, i, axis=1)
                nn_reduced = self.train(X_reduced, y_train)
                loss = nn_reduced.loss_
                score = (baseline_loss - loss) / baseline_loss
                scores.append(score)
            scores = np.array(scores)
        else:
            raise ValueError('Invalid importance method specified')
        return scores

    def train(self, x, y):
        return self.nn.fit(x,y)

    def feature_selection(self, X_train, y_train, n_features_to_select):
        """
        Perform recursive feature elimination to select a subset of the input features based on
        their importance scores.

        Parameters:
        - X_train: array-like of shape (n_samples, n_features), the input data for training 
        - y_train: array-like of shape (n_samples,), the target variable for training 
        - n_features_to_select: int, the number of features to select
        Returns:
	    - X_train_reduced: array-like of shape (n_samples, n_features_to_select), the input data
	      with selected features
	    """
        self.eliminated_feature_indices = []
        selected_feature_indices = np.arange(X_train.shape[1])
        
		# Recursively select features until n_features_to_select are selected 
        while X_train.shape[1] > n_features_to_select:
            # Compute importance scores 
            scores = self._get_importance_scores(X_train, y_train) 
            # Select top n_features_to_select features 
            worst_feature_index = np.argsort(scores)[0] 
            self.eliminated_feature_indices.append(worst_feature_index)
            X_train = np.delete(X_train, worst_feature_index, axis=1)
            self.n_features = X_train.shape[1]
            
        # Get the indices of the selected features
        for i in range(len(self.eliminated_feature_indices)):
            selected_feature_indices = np.delete(selected_feature_indices, self.eliminated_feature_indices[i])
        self.selected_feature_indices = selected_feature_indices
        return X_train