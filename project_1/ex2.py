# Using much code from week 37 slides
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from ex1 import OLS, create_X
from franke import FrankeFunction

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(2018)
n_boostraps = 100
maxdegree = 14
N = 50 # number of samples
e = 0 # noise weight

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
error = np.zeros(maxdegree)
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
    z_pred = np.empty((z_test.shape[0], n_boostraps)) #??
    print(z_test.shape)
    print(z_pred.shape)
    for i in range(n_boostraps):
        X_, z_ = resample(X_train, z_train)
        beta = OLS(X_, z_)
        z_pred[:, i] = (X_test @ beta) # ??

    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    print('Polynomial degree:', degree)
    print('Error:', error[degree])
    print('Bias^2:', bias[degree])
    print('Var:', variance[degree])
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
#plt.show()