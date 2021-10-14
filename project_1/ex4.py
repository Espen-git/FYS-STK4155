"""
Much the same code as in ex2.py and ex3.py but using
Ridge insead of OLS
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random
from sklearn.model_selection import train_test_split
from ex1 import MSE, create_X
from ex2 import resample
from ex3 import cross_validation_split
from franke import FrankeFunction
from sklearn.preprocessing import StandardScaler

def Ridge(X, z, l):
    """
    x: Data matrix
    z: Target values
    l: lambda

    beta: Solution to OLS
    """
    beta = np.linalg.pinv(X.T.dot(X)+l * np.identity(X.shape[1])).dot(X.T).dot(z)
    return beta

def Surface_plot(data, lambdas, maxdegree, name):
    var_x = lambdas
    var_y = np.arange(maxdegree)
    var_x, var_y = np.meshgrid(var_x, var_y)
    #fig = plt.figure(figsize=(32,12))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(var_x, var_y, data, cmap=cm.plasma, linewidth = 0, antialiased=False)
    ax.set_xlabel('Lambda values')
    ax.set_ylabel('Polynomial degree')
    ax.set_zlabel('Value')
    plt.savefig(name, dpi=300)
    fig.show()

def ex4_bootstrap():
    # Bootstarp as ex2.py but using Ridge
    n_bootstraps = 50
    maxdegree = 10
    N = 40 # number of samples
    e = 0.1 # noise weight

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)

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

    # Decide which values of lambda to use
    nlambdas = 5
    MSE_test_lambda = np.zeros((nlambdas, maxdegree))
    MSE_training_lambda = np.zeros((nlambdas, maxdegree))
    bias_lambda = np.zeros((nlambdas, maxdegree))
    variance_lambda = np.zeros((nlambdas, maxdegree))
    lambdas = np.logspace(-4, 1, nlambdas)
    for l in range(nlambdas):
        lmb = lambdas[l]
        mse_test = np.zeros(maxdegree)
        mse_training = np.zeros(maxdegree)
        bias = np.zeros(maxdegree)
        variance = np.zeros(maxdegree)
        for degree in range(maxdegree):
            z_tilde = np.zeros((len(z_train), n_bootstraps))
            z_predict = np.zeros((len(z_test), n_bootstraps))
            testing_error= []
            training_error = []
            for i in range(n_bootstraps):
                #X_, z_ = resample(X_train, z_train)
                x_and_y_train_resampled, z_ = resample(x_and_y_train, z_train)

                # Scale training data
                x_and_y_train_scaled = scaler.transform(x_and_y_train_resampled)

                X_train = create_X(x_and_y_train_scaled.T[0], x_and_y_train_scaled.T[1], degree+1)
                X_test = create_X(x_and_y_test_scaled.T[0], x_and_y_test_scaled.T[1], degree+1)

                beta = Ridge(X_train, z_, lmb)
                z_predict_train = (X_train @ beta).ravel()
                z_predict_test = (X_test @ beta).ravel()
                z_tilde[:,i] = z_predict_train
                z_predict[:,i] = z_predict_test

                testing_error.append(MSE(z_test, z_predict_test))
                training_error.append(MSE(z_, z_predict_train))

            #polydegree.append(degree)
            mse_test[degree] = np.mean(testing_error)
            mse_training[degree] = np.mean(training_error)
            bias[degree] = np.mean( (z_test.reshape(z_test.shape[0],1) - np.mean(z_predict, axis=1, keepdims=True))**2 )
            variance[degree] = np.mean( np.var(z_predict, axis=1, keepdims=True) )

            if show_print:
                print('Lambda:', lambdas[l])
                print('Polynomial degree:', degree+1)
                print('Error:', mse_test[degree])
                print('Bias^2:', bias[degree])
                print('Var:', variance[degree])
                print('{} >= {} + {} = {}'.format(mse_test[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))
        
        MSE_test_lambda[l,:] = mse_test
        MSE_training_lambda[l,:] = mse_training
        bias_lambda[l,:] = bias
        variance_lambda[l,:] = variance

    Surface_plot(MSE_training_lambda.T, lambdas, maxdegree, "ex4_traningerror_bootstrap.png")
    Surface_plot(MSE_test_lambda.T, lambdas, maxdegree, "ex4_testerror_bootstrap.png")
    Surface_plot(bias_lambda.T, lambdas, maxdegree, "ex4_bias_bootstrap.png")
    Surface_plot(variance_lambda.T, lambdas, maxdegree, "ex4_variance_bootstrap.png")

def ex4_cross_validation():
    # Cross-validation as ex3.py but using Ridge    
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

    # Decide which values of lambda to use
    nlambdas = 5
    MSE_test_lambda = np.zeros((nlambdas, maxdegree))
    MSE_training_lambda = np.zeros((nlambdas, maxdegree))
    bias_lambda = np.zeros((nlambdas, maxdegree))
    variance_lambda = np.zeros((nlambdas, maxdegree))
    lambdas = np.logspace(-4, 1, nlambdas)
    for l in range(nlambdas):
        lmb = lambdas[l]
        mse_test = np.zeros(maxdegree)
        mse_training = np.zeros(maxdegree)
        bias = np.zeros(maxdegree)
        variance = np.zeros(maxdegree)

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
                beta = Ridge(X_train, z_train, lmb)
                
                # Do predictions
                z_pred_test = (X_test @ beta)
                z_predict[:,i] = z_pred_test
                z_pred_train = (X_train @ beta)
                z_tilde[:,i] = z_pred_train
            
            mse_training[degree] = np.mean( np.mean((z_train.reshape(z_train.shape[0],1) - z_tilde)**2, axis=1, keepdims=True) )   
            mse_test[degree] = np.mean( np.mean((z_test.reshape(z_test.shape[0],1) - z_predict)**2, axis=1, keepdims=True) )
            bias[degree] = np.mean( (z_test.reshape(z_test.shape[0],1) - np.mean(z_predict, axis=1, keepdims=True))**2 )
            variance[degree] = np.mean( np.var(z_tilde, axis=1, keepdims=True) )
            
            if show_print:
                print('Lambda:', lambdas[l])
                print('Polynomial degree:', degree+1)
                print('Error:', mse_test[degree])
                print('Bias^2:', bias[degree])
                print('Var:', variance[degree])
                print('{} >= {} + {} = {}'.format(mse_test[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

        MSE_test_lambda[l,:] = mse_test
        MSE_training_lambda[l,:] = mse_training
        bias_lambda[l,:] = bias
        variance_lambda[l,:] = variance

    Surface_plot(MSE_training_lambda.T, lambdas, maxdegree, "ex4_traningerror_cv.png")
    Surface_plot(MSE_test_lambda.T, lambdas, maxdegree, "ex4_testerror_cv.png")
    Surface_plot(bias_lambda.T, lambdas, maxdegree, "ex4_bias_cv.png")
    Surface_plot(variance_lambda.T, lambdas, maxdegree, "ex4_variance_cv.png")

if __name__ == "__main__":
    np.random.seed(2021)
    show_print = False

    ex4_bootstrap()
    ex4_cross_validation()