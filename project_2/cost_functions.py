import functions
import numpy as np

class MSE():
    def __init__(self):
        self.name = "MSE"

    def __call__(self, z_pred, z_real):
        return np.sum((z_pred - z_real)**2) / np.size(z_real)

    def derivative(self, z_pred, z_real):
        return 2 * (z_pred - z_real) / z_real.shape[0] # kan dette være np.size(z_real)

class CrossEntropy():
    def __init__(self):
        self.name = "CrossEntropy"
    
    def __call__(self, z_pred, z_real):
        return -np.log(np.prod(np.pow(z_pred, z_real)))

    def derivative(self, z_pred, z_real):
        return z_pred - z_real