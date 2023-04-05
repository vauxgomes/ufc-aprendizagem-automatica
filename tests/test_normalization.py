# Author: Vaux Gomes
# Contact: vauxgomes@gmail.com
# Version: 0.1

'''
Description
-----------
Load peixe dataset and apply 
normalization on the data
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

#
# Import data
data = np.genfromtxt('./data/peixe.txt', delimiter=',')

# Normalizer
normalizer = ZScore()

# Normalization
data_ = normalizer.fit_transform(data)

# Restoring
print(normalizer.restore(data_))