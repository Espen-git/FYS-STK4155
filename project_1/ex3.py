import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split
from ex1 import OLS, create_X
from franke import FrankeFunction
from ex1 import OLS, MSE, R2, create_X
from franke import FrankeFunction

def cross_validation_split(X, z, folds=10):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

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
n_bootstraps = 50
maxdegree = 10
N = 40 # number of samples
e = 0.1 # noise weight

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
mse_test = np.zeros(maxdegree)
mse_training = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

z = FrankeFunction(xx, yy) # True values
# add noise
noise = np.random.randn(N,N) # NxN matrix of normaly distributed noise
z = z + e*noise

# Stacking x and y and split data into test and train
x_and_y=np.hstack((xx.ravel().reshape(xx.ravel().shape[0],1),yy.ravel().reshape(yy.ravel().shape[0],1)))
x_and_y_train, x_and_y_test, z_train, z_test = train_test_split(x_and_y,z.ravel(),test_size=0.2, random_state=2021, shuffle=True)

# Scaling data
scaler = StandardScaler()
scaler.fit(x_and_y_test)
# Scale test data. will scale traning data later
x_and_y_test_scaled = scaler.transform(x_and_y_test) 

for degree in range(maxdegree):
    """
    # Scaling data
    scaler = StandardScaler() # Chose scaling method
    scaler.fit(x_and_y_train) # fit scaler
    x_and_y_train_scaled = scaler.transform(x_and_y_train) # Scale training data
    x_and_y_test_scaled = scaler.transform(x_and_y_test) # Scale test data

    X_train = create_X(x_and_y_train_scaled.T[0], x_and_y_train_scaled.T[1], degree)
    X_test = create_X(x_and_y_test_scaled.T[0], x_and_y_test_scaled.T[1], degree)
    """
    z_tilde = np.zeros((len(z_train), n_bootstraps))
    z_predict = np.zeros((len(z_test), n_bootstraps))
    testing_error= []
    training_error = []
    """
    for i in range(n_bootstraps):
        #X_, z_ = resample(X_train, z_train)
        x_and_y_train_resampled, z_ = resample(x_and_y_train, z_train)

        # Scale training data
        x_and_y_train_scaled = scaler.transform(x_and_y_train_resampled)

        X_train = create_X(x_and_y_train_scaled.T[0], x_and_y_train_scaled.T[1], degree)
        X_test = create_X(x_and_y_test_scaled.T[0], x_and_y_test_scaled.T[1], degree)

        beta = OLS(X_train, z_)
        z_predict_train = (X_train @ beta).ravel()
        z_predict_test = (X_test @ beta).ravel()
        z_tilde[:,i] = z_predict_train
        z_predict[:,i] = z_predict_test

        testing_error.append(MSE(z_test, z_predict_test))
        training_error.append(MSE(z_, z_predict_train))

    polydegree[degree] = degree
    mse_test[degree] = np.mean(testing_error)
    mse_training[degree] = np.mean(training_error)
    bias[degree] = np.mean( (z_test.reshape(z_test.shape[0],1) - np.mean(z_predict, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(z_predict, axis=1, keepdims=True) )

    print('Polynomial degree:', degree)
    print('Error:', mse_test[degree])
    print('Bias^2:', bias[degree])
    print('Var:', variance[degree])
    print('{} >= {} + {} = {}'.format(mse_test[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))
    """
"""
plt.plot(polydegree, mse_test, label='Test error')
plt.plot(polydegree, mse_training, label='Traning error')
plt.plot(polydegree, bias, label="Bias", linestyle="--")
plt.plot(polydegree, variance, label="Variance", linestyle="--")
plt.xticks([0,1,2,3,4,5,6,7,8,9],["1","2","3","4","5","6","7","8","9","10"])
plt.xlabel("polynomial degree")
plt.legend()
plt.show()
"""