import random

import numpy as np
from scipy import spatial
from scipy import stats

class KNN:
    """
    Implementation of the k-nearest neighbors algorithm for classification
    and regression problems.
    """
    def __init__(self, k, aggregation_function):
        """
        Takes two parameters.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point. The
        aggregation_function is either "mode" for classification or
        "average" for regression.
        """
        self.k = k
        if aggregation_function == 'mode' or 'average':
            self.aggregation_function = aggregation_function
        else:
            raise ValueError('aggregation_function must be either mode or average')
            
        self.points = None
        self.labels = None
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        """
        self.points = X
        self.labels = y
        
    def predict(self, X):
        """
        Predicts the output variable's values for the query points X.
        """
        distances = spatial.distance.cdist(X, self.points)
        closest = np.argsort(distances, axis=1).transpose()[:self.k].transpose()
        labels = self.labels[closest]
        if self.aggregation_function == 'mode':
            return stats.mode(labels, axis=1).mode.flatten()
        elif self.aggregation_function == 'average':
            return np.mean(labels, 1)
        
        