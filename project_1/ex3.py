import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from ex1 import OLS, create_X
from franke import FrankeFunction
from ex1 import OLS, MSE, R2, create_X

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

if __name__ == "__main__":
    np.random.seed(2021)
    maxdegree = 10
    N = 40 # number of samples
    k_folds = 10 # number of k-folds

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)
    mse_test = np.zeros(maxdegree)
    mse_training = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)

    z = FrankeFunction(xx, yy) # True values
    # add noise
    e = 0.1 # noise weight
    noise = np.random.randn(N,N) # NxN matrix of normaly distributed noise
    z = z + e*noise

    # Stacking x and y
    x_and_y = np.hstack((xx.ravel().reshape(xx.ravel().shape[0],1),yy.ravel().reshape(yy.ravel().shape[0],1)))

    # Scaling data
    scaler = StandardScaler()
    scaler.fit(x_and_y)
    x_and_y_scaled = scaler.transform(x_and_y) 

    for degree in range(maxdegree):
        z_predict = np.zeros((int(z.ravel().shape[0]*(1/k_folds)), k_folds))
        z_tilde = np.zeros((int(z.ravel().shape[0]*(1-(1/k_folds))), k_folds))

        data = np.hstack((x_and_y_scaled,z.ravel().reshape(N**2,1)))
        
        # Make folds 
        folds = cross_validation_split(data, k_folds)
        for i in range(k_folds):
            # Make train and test data using the i'th fold
            n_fold = folds.copy()
            test_data = n_fold.pop(i)
            test_data = np.asarray(test_data)
            train_data = np.vstack(n_fold)
            
            # split X and z
            z_train = train_data[:,-1]
            xy_train = train_data[:,0:-1]
            z_test = test_data[:,-1]
            xy_test = test_data[:,0:-1]

            # Calculate beta
            X_train = create_X(xy_train.T[0], xy_train.T[1], degree+1)
            X_test = create_X(xy_test.T[0], xy_test.T[1], degree+1)
            beta = OLS(X_train, z_train)
            
            # Do predictions
            z_pred_test = (X_test @ beta)
            z_predict[:,i] = z_pred_test
            z_pred_train = (X_train @ beta)
            z_tilde[:,i] = z_pred_train
        
        mse_training[degree] = np.mean( np.mean((z_train.reshape(z_train.shape[0],1) - z_tilde)**2, axis=1, keepdims=True) )   
        mse_test[degree] = np.mean( np.mean((z_test.reshape(z_test.shape[0],1) - z_predict)**2, axis=1, keepdims=True) )
        bias[degree] = np.mean( (z_test.reshape(z_test.shape[0],1) - np.mean(z_predict, axis=1, keepdims=True))**2 )
        variance[degree] = np.mean( np.var(z_tilde, axis=1, keepdims=True) )
        
        print('Polynomial degree:', degree+1)
        print('Error:', mse_test[degree])
        print('Bias^2:', bias[degree])
        print('Var:', variance[degree])
        print('{} >= {} + {} = {}'.format(mse_test[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))
        
    plt.figure()
    plt.plot(mse_training, label="MSE train")
    plt.plot(mse_test, label="MSE test")
    plt.legend()
    plt.xticks([0,1,2,3,4,5,6,7,8,9],["1","2","3","4","5","6","7","8","9","10"])
    plt.xlabel("polynomial degree")
    plt.ylabel("Error")
    plt.show()

    plt.figure()
    plt.plot(bias, label="Bias")
    plt.plot(bias+variance, label="Bias+Variance")
    plt.plot(variance, label="Variance")
    plt.plot(mse_test, label="MSE test")
    plt.legend()
    plt.xticks([0,1,2,3,4,5,6,7,8,9],["1","2","3","4","5","6","7","8","9","10"])
    plt.xlabel("polynomial degree")
    plt.show()