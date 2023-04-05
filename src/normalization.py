import numpy as np

#
def norm_z(data, noise=10**(-10), verbatin=False):
    ''' Função de normalização escore-Z'''

    # Dimension
    N = data.shape[0]

    # Mean
    mean = data.mean(axis=0)
    
    # Mean difference
    diff = data - mean

    # Standard Deviation
    std = np.sqrt(1/(N-1) * (diff**2).sum(axis=0))
    
    # Normalized data
    norm_data = diff/(std + noise)
    
    # Data information
    if verbatin:
        print(f'# Noise: {noise}')
        print(f'# Data shape: {data.shape} \n')
        print(f'# Data head:\n{data[:5,:]} \n')
        print(f'# Standard deviation:\n{std} \n')
        print(f'# Normalized data head:\n{norm_data[:5,:]} \n')
        
    return norm_data, mean, std

def denorm_z(norm_data, mean, std):
    ''' Função de desnormalização escore-Z'''
    return std*norm_data + mean

if __name__ == '__main__':
  # Import data
  data = np.genfromtxt('./data/peixe.txt', delimiter=',')

  # Normalization
  norm_data, mean, std = norm_z(data, verbatin=True)

  # Denormalization
  print(denorm_z(norm_data, mean, std))