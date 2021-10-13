import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from ex1 import OLS, create_X
from franke import FrankeFunction

def k_fold_OLS_my_code(X, z, k=5):
    """
    k - Number of folds
    """
    pass

def k_fold_OLS_sklearn(X, z, k=5):
    """
    k - Number of folds
    """
    kfold = KFold(n_splits = k)
    scores_KFold = np.zeros(k)

    reg = LinearRegression()
    i = 0

    for train_inds, test_inds in kfold.split(X):
        xtrain = X[train_inds]
        ztrain = z[train_inds, :]

        xtest = X[test_inds]
        ztest = z[test_inds]

        reg.fit(xtrain, ztrain)
        zpred = reg.predict(xtest)
        
        
        #scores_KFold[i] = np.sum
        i += 1

np.random.seed(2021)
N = 100 # number of samples
n = 5
e = 0.1 # noise weight

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
z = FrankeFunction(xx, yy) # True values
# add noise
noise = np.random.randn(N,N) # NxN matrix of normaly distributed noise
z = z + e*noise

X = create_X(x, y, n) # Creates the design matrix

"""
X_train, X_test, z_train, z_test = train_test_split(X, z, 
    test_size=0.2, random_state=2021)

# Scale data
scaler = StandardScaler() # Chose scaling method
scaler.fit(X_train) # fit scaler
X_train = scaler.transform(X_train) # Scale training data
X_test = scaler.transform(X_test) # Scale test data 
# Then the same for targets
scaler = StandardScaler()
scaler.fit(z_train)
z_train = scaler.transform(z_train)
z_test = scaler.transform(z_test)
"""

k_fold_OLS_sklearn(X, z)