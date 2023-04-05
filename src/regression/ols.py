# Author: Vaux Gomes
# Contact: vauxgomes@gmail.com
# Version: 0.1

'''
Description
-----------
This module implements a Linear Regression via 
Ordinary Least Squares (OLS)
'''

# Imports
import numpy as np
from sklearn.base import \
    BaseEstimator, RegressorMixin

#
class OrdinaryLeastSquares(BaseEstimator, RegressorMixin):
    ''' Linear Regression via Ordinary Least Squares (OLS) '''

    #
    def __init__(self, noise=10**(-15)):
        self.noise = noise

    #
    def fit(self, X, y):
        # Auxiliary
        N = X.shape[0]

        # Adding a column of ones and some noise
        X_ = np.concatenate((np.ones((N,1)), X), axis=1) + self.noise

        # Calculating ŵ
        self.w_hat = np.linalg.inv(X_.T @ X_) @ X_.T @ y

    #
    def predict(self, X, y=None):
        # Auxiliary
        N = X.shape[0]

        # Adding a column of ones
        X_ = np.concatenate((np.ones((N,1)), X), axis=1)

        # Predictions
        return X_ @ self.w_hat
    
    #
    def __str__(self):
        return f'Ordinary Least Squares: {self.w_hat}'