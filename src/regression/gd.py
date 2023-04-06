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
from src.metrics.metrics import rmse

#
class LinearRegressionGD(BaseEstimator, RegressorMixin):
    ''' Linear Regression via Gradient Decent (GD) '''

    #
    def __init__(self, alpha=10**-5, max_generations = 10**5):
        self.max_generations = max_generations
        self.alpha = alpha

    #
    def fit(self, X, y, verbatim=False):
        # Auxiliary
        M = X.shape[1]

        # Initial weights
        self.w = np.zeros(M)

        # Initial Error
        err = rmse(y, X @ self.w) 

        # Main loop
        for i in range(self.max_generations):
            # Step
            w_ = self.w + self.alpha * (y - self.w.T * X).mean(axis=0)

            # Step error
            err_ = rmse(y, X @ w_)

            # 
            if err < err_:
                if verbatim:
                    print('-'*20)
                    print(f'Iteration: {i:>8}')
                    print(f'Error: {err_:>12.5}')
                    print('-'*20)

                
                break
            
            err = err_
            self.w = w_

    def predict(self, X, y=None):
        return X @ self.w
    
    #
    def __str__(self):
        return f'Linear Regression via Gradient Decent: {self.w}'