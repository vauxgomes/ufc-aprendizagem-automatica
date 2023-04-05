# Author: Vaux Gomes
# Contact: vauxgomes@gmail.com
# Version: 0.1

'''
Description
-----------
For this example we wanted to train 
and test on the same data
''' 

# Imports
import os
import sys
import numpy as np

# Fixing paths
sys.path.append(os.path.join(
    os.path.dirname(__file__), '../'))

# Local imports
from src.normalization.zscore import ZScore
from src.metrics.metrics import rmse
from src.regression.ols import OrdinaryLeastSquares

#
# Import data
data = np.genfromtxt('./data/peixe.txt', delimiter=',')

# Separation
X = data[:,:-1]
y = data[:, -1]

# Normalization
normalizer = ZScore()
X = normalizer.fit_transform(X)

# Model training
ols = OrdinaryLeastSquares()
ols.fit(X, y)

# print(ols)

# Predict
y_hat = ols.predict(X)

# Test error
print(f'RMSE: {rmse(y, y_hat)}')
