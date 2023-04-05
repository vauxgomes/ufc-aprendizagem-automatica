# Author: Vaux Gomes
# Contact: vauxgomes@gmail.com
# Version: 0.1

'''
Description
-----------
Implementation of Z-score normalization
'''

# Imports
import numpy as np

#
class ZScore():
    ''' Z-score Normalizer '''

    #
    def __init__(self, noise=10**(-15)):
        self.noise = noise

    #
    def fit_transform(self, X, verbatim=False):
        ''' Z-score normalization '''

        # Dimension
        N = X.shape[0]

        # Mean
        self.mean = X.mean(axis=0)
        
        # Mean difference
        diff = X - self.mean

        # Standard Deviation
        self.std = np.sqrt(1/(N-1) * (diff**2).sum(axis=0))
        
        # Normalized data
        X_ = diff/(self.std + self.noise)
        
        # Data information
        if verbatim:
            print(f'# Noise: {self.noise}')
            print(f'# Data shape: {X.shape} \n')
            print(f'# Data head:\n{X[:5,:]} \n')
            print(f'# Standard deviation:\n{self.std} \n')
            print(f'# Normalized data head:\n{X_[:5,:]} \n')
            
        return X_
    
    #
    def restore(self, X):
        return self.std * X + self.mean