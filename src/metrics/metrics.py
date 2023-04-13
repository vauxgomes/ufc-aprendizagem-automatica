# Author: Vaux Gomes
# Contact: vauxgomes@gmail.com
# Version: 0.1

'''
Description
-----------
This module implements all necessary 
metrics for this project
'''

# Imports
import numpy as np

def mse(y, y_hat):
  ''' Mean squared error '''
  return ((y - y_hat)**2).mean()

def rmse(y, y_hat):
  ''' Root mean squared error '''
  return np.sqrt(((y - y_hat)**2).mean())