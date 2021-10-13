# Using much code from week 37 slides
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from ex1 import OLS, MSE, R2, create_X
from franke import FrankeFunction
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(2018)
n_bootstraps = 100
maxdegree = 14
N = 100 # number of samples
e = 0.15 # noise weight

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
error_test = np.zeros(maxdegree)
error_training = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

z = FrankeFunction(xx, yy) # True values
# add noise
np.random.seed(2021)
noise = np.random.randn(N,N) # NxN matrix of normaly distributed noise
z = z + e*noise

for degree in range(maxdegree):
    X = create_X(x, y, degree) # Creates the design matrix
    X_train, X_test, z_train, z_test = train_test_split(X, z, 
        test_size=0.2, random_state=2021)
    z_pred = np.empty((z_test.shape[0], z_test.shape[1], n_bootstraps))
    z_tilde = np.empty((z_train.shape[0], z_train.shape[1], n_bootstraps))
    
    scaler = StandardScaler() # Chose scaling method
    scaler.fit(X_train) # fit scaler
    X_train = scaler.transform(X_train) # Scale training data
    X_test = scaler.transform(X_test) # Scale test data 
    # Then the same for targets
    scaler = StandardScaler()
    scaler.fit(z_train)
    z_train = scaler.transform(z_train)
    z_test = scaler.transform(z_test)
    
    for i in range(n_bootstraps):
        X_, z_ = resample(X_train, z_train)
        beta = OLS(X_, z_)
        z_pred[:, :, i] = (X_test @ beta)
        z_tilde[:,:,i] = (X_train @ beta)

    polydegree[degree] = degree
    
    
    error_bootstraps_test = np.zeros(z_pred.shape)
    error_bootstraps_traning = np.zeros(z_tilde.shape)
    for i in range(n_bootstraps):
        error_bootstraps_test[:,:,i] =  (z_test - z_pred[:,:,i])**2 # error for each bootstrap
        error_bootstraps_traning[:,:,i] =  (z_train - z_tilde[:,:,i])**2 # error for each bootstrap
    error_test[degree] = np.mean(np.mean(error_bootstraps_test, axis=2))
    error_training[degree] = np.mean(np.mean(error_bootstraps_traning, axis=2))
    
    bias[degree] = np.mean((z_test - np.mean(z_pred, axis=2))**2)
    variance[degree] = np.mean(np.var(z_pred, axis=2))
    #print('Polynomial degree:', degree)
    #print('Error:', error_test[degree])
    #print('Bias^2:', bias[degree])
    #print('Var:', variance[degree])
    #print('{} >= {} + {} = {}'.format(error_test[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.plot(polydegree, error_test, label='Test error')
plt.plot(polydegree, error_training, label='Traning error')
#plt.plot(polydegree, bias, label="Bias")
#plt.plot(polydegree, variance, label="Variance")
plt.legend()
plt.show()