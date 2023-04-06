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
from src.regression.gd import LinearRegressionGD

#
# Import data
data = np.genfromtxt('./data/peixe.txt', delimiter=',')

# Normalization
normalizer = ZScore()
data = normalizer.fit_transform(data)

# Separation
X = data[:, :-1]
y = data[:, -1:]

# Model training
model = LinearRegressionGD()
model.fit(X, y, verbatim=True)

# Predict
y_hat = model.predict(X)

# Test error
print(f'RMSE: {rmse(y, y_hat):>13.5}')