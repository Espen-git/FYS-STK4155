from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def FrankeFunction(x,y):
    """
    Calculates the Franke function for given x,y values
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_data(N, e, seed=1337):
    """
    Crates x,y data ad well as z calculated from Franke function on
    the x,y data.

    N - number of samples
    e - noise error
    """
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y) # True values
    # add noise
    np.random.seed(seed)
    noise = np.random.randn(N,N) # NxN matrix of normaly distributed noise
    z = z + e*noise

    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    return x, y, z

def create_X(x, y, n):
    """
    n - Polynomial degree
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in theta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def MSE(z_actual, z_computed):
    n = len(z_actual)
    mse = np.sum((z_actual - z_computed) ** 2 ) / n
    return mse

def R2(z_actual, z_computed):
    numerator = 1 - np.sum((z_actual - z_computed) ** 2)
    denominator = np.sum((z_actual - np.mean(z_actual)) ** 2)
    r2 = numerator / denominator
    return r2

def OLS(X, z):
    """
    x: Data matrix
    z: Target values

    beta: Solution to OLS
    """
    theta = np.linalg.pinv(X.T @ X) @ X.T @ z
    return theta

def Ridge(X, z, lmb):
    """
    x: Data matrix
    z: Target values
    l: lambda

    beta: Solution to OLS
    """
    I = np.eye(X.shape[1], X.shape[1])
    theta = np.linalg.pinv(X.T.dot(X) + lmd * I).dot(X.T).dot(z)
    return theta

def Acc(z_actual, z_computed):
    return np.mean(z_actual==z_computed)

def make_data_ready_for_regression(X, z, n=5):
    """
    X - data from create_X function 
    z - target data from create_data function
    n - polynomial degree
    """
    X_train, X_test, z_train, z_test = train_test_split(X, z, 
            test_size=0.2, random_state=1337, shuffle=True)

    # Scaling data
    X_scaler = StandardScaler() # Chose scaling method
    X_scaler.fit(X_train) 
    X_train_scaled = X_scaler.transform(X_train) # Scale traning data
    X_test_scaled = X_scaler.transform(X_test)# Scale test data

    z_scaler = StandardScaler()
    z_scaler.fit(z_train.reshape(-1,1))
    z_train_scaled = z_scaler.transform(z_train.reshape(-1,1))
    z_test_scaled = z_scaler.transform(z_test.reshape(-1,1))

    return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled

def resample(X, y):
    samples = len(y)
    resampled_X = np.zeros((X.shape))
    resampled_y = np.zeros((y.shape))
    for i in range(samples):
        random_index = random.randint(0, samples-1)
        resampled_X[i,:] = X[random_index, :]
        resampled_y[i] = y[random_index]
    return resampled_X, resampled_y

def cross_validation_split(dataset, folds=10):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split