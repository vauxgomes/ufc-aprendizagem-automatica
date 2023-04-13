# Author: Vaux Gomes
# Contact: vauxgomes@gmail.com
# Version: 0.1

'''
Description
-----------
This module implements a Linear Regression via 
Gradient Decent (GD)
'''

# Imports
import numpy as np
from sklearn.base import \
  BaseEstimator, RegressorMixin

# Local imports
from src.metrics.metrics import mse

#
class LinearRegressionGD(BaseEstimator, RegressorMixin):
    ''' Linear Regression via Gradient Decent (GD) '''

    #
    def __init__(self, alpha=10**-3, max_iter = 10**5):
        self.max_iter = max_iter
        self.alpha = alpha

    #
    def fit(self, X, y, verbatim=False):
        # Cleaning
        self.costs = []

        # Auxiliary
        _, m = X.shape

        # Initial weights
        self.w = np.zeros(m)

        # Initial Error
        self.costs.append(np.inf)

        # Main loop
        for i in range(self.max_iter):
            # Step
            w_ = self.w + self.alpha * ((y - self.w.T * X) * X).sum(axis=0)

            # Step error
            err = mse(y, X @ w_)

            # 
            if self.costs[-1] <= err:
                if verbatim:
                    print('-'*20)
                    print(f'Iteration: {i:>8}')
                    print(f'Error: {err:>12.5}')
                    print('-'*20)
                
                break
            else:
                print(err)
            
            self.costs.append(err)
            self.w = w_

    def predict(self, X, y=None):
        return X @ self.w
    
    #
    def __str__(self):
        return f'Linear Regression via Gradient Decent: {self.w}'