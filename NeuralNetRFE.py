# -*- coding: utf-8 -*-
"""
Trevor Amestoy
Cornell University


"""

import numpy as np
from sklearn.neural_network import MLPRegressor

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

class NeuralNetRFE:
    def __init__(self,
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
        self.importance = importance

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
            self.nn.fit(X_train, y_train)
            scores = np.abs(self.nn.coefs_[0]).sum(axis=1)
            return scores
        else:
            raise ValueError('Other importance metrics have not been implemented.')

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
        self.n_features = X_train.shape[1]
        self.eliminated_feature_indices = []
        selected_feature_indices = np.arange(X_train.shape[1])

		# Recursively select features until n_features_to_select are selected
        while X_train.shape[1] > n_features_to_select:
            # Compute importance scores
            scores = self._get_importance_scores(X_train, y_train)

            # Select top n_features_to_select
            worst_feature_index = np.argsort(scores)[0]
            self.eliminated_feature_indices.append(worst_feature_index)
            X_train = np.delete(X_train, worst_feature_index, axis=1)
            self.n_features = X_train.shape[1]

        # Get the indices of the selected features
        for i in range(len(self.eliminated_feature_indices)):
            selected_feature_indices = np.delete(selected_feature_indices, self.eliminated_feature_indices[i])
        self.selected_feature_indices = selected_feature_indices
        return X_train
