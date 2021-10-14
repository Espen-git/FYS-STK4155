import numpy as np
import random
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from ex1 import OLS, MSE, R2, create_X
from ex3 import cross_validation_split
from ex4 import Surface_plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def reg1(x_and_y, z):
    maxdegree = 10
    k_folds = 10 # number of k-folds

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

def reg2():
    pass

def reg3():
    pass

if __name__ == "__main__":
    # Load the terrain
    terrain = imread('SRTM_data_Norway_1.tif')
    show_terrain = False
    if show_terrain:
        # Show the terrain
        plt.figure()
        plt.title('Terrain over Norway 1')
        plt.imshow(terrain, cmap='gray')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    z = terrain

    x = np.asarray(np.arange(z.shape[0]))
    y = np.asarray(np.arange(z.shape[1]))
    xx, yy = np.meshgrid(x, y)

    x_and_y = np.hstack((xx.ravel().reshape(xx.ravel().shape[0],1),yy.ravel().reshape(yy.ravel().shape[0],1)))

    reg1(x_and_y, z)