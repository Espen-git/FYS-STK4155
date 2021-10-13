from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from franke import FrankeFunction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Much used from week35 slides
# And some inspiation from https://www.kaggle.com/bjoernjostein/linear-polynomial-fitting-of-franke-s-function
def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def OLS(X, z):
    """
    x: Data matrix
    z: Target values

    beta: Solution to OLS
    """
    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)
    return beta

def MSE(z_actual, z_comuted):
    mse = 0
    n = len(z_actual)
    for z in zip(z_actual, z_comuted):
        mse += ((z[0] - z[1])**2)
    mse = mse / n
    return mse

def R2(z_actual, z_comuted):
    z_mean = np.mean(z_actual)
    sum1 = 0
    sum2 = 0
    for z in zip(z_actual, z_comuted):
        sum1 += ((z[0] - z[1])**2)
        sum2 += ((z[0] - z_mean)**2)
    r2 = 1 - (sum1/sum2)
    return r2

if __name__ == "__main__":
    n = 20 # polynomial degree
    N = 50 # number of samples
    e = 0.1 # noise weight
    show_beta_plot = True
    show_plot = True

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)

    z = FrankeFunction(xx, yy) # True values
    # add noise
    np.random.seed(2021)
    noise = np.random.randn(N,N) # NxN matrix of normaly distributed noise
    z = z + e*noise

    # Stacking x and y and split data into test and train
    x_and_y=np.hstack((xx.ravel().reshape(xx.ravel().shape[0],1),yy.ravel().reshape(yy.ravel().shape[0],1)))
    x_and_y_train, x_and_y_test, z_train, z_test = train_test_split(x_and_y,z.ravel(),test_size=0.2, random_state=2021, shuffle=True)

    # Scaling data
    scaler = StandardScaler() # Chose scaling method
    scaler.fit(x_and_y_train) # fit scaler
    x_and_y_train_scaled = scaler.transform(x_and_y_train) # Scale training data
    x_and_y_test_scaled = scaler.transform(x_and_y_test) # Scale test data

    X_train = create_X(x_and_y_train_scaled.T[0], x_and_y_train_scaled.T[1], n)
    X_test = create_X(x_and_y_test_scaled.T[0], x_and_y_test_scaled.T[1], n)
    beta = OLS(X_train, z_train)
    z_tilde = X_train @ beta
    z_predict = X_test @ beta
    
    # Compute confidence intervals for beta
    hessian_inverted = np.linalg.pinv(X_train.T.dot(X_train))
    beta_sigma = np.sqrt(hessian_inverted.diagonal()) # standard deviation for each beta
    z_value = 1.96 # Confidence of 95%
    confidence_intervals = np.zeros([len(beta)])
    for i in range(len(beta)):
        confidence_intervals[i] = (z_value * beta_sigma[i]) / np.sqrt(N)

    if show_beta_plot:
        plt.errorbar(range(len(beta)), beta, yerr=confidence_intervals, fmt='.')
        plt.xlabel('i')
        plt.ylabel('beta_i')
        plt.title('95% Confidence intervals of beta_i')
        plt.show()
    
    # MSE
    # Scikit-learns MSE function
    mse_train = MSE(z_train, z_tilde)
    mse_test = MSE(z_test, z_predict)
    print(f"MSE train: {mse_train}")
    print(f"MSE test: {mse_test}")

    #R^2
    r2_train = R2(z_train, z_tilde)
    r2_test = R2(z_test, z_predict)
    print(f"R2 train:{r2_train}")
    print(f"R2 test:{r2_test}")

    if show_plot:
        # plotting prediction based on all data
        X_plot = create_X(xx, yy, n)
        beta = OLS(X_plot, z.ravel())
        z_plot = X_plot @ beta
        z_plot = z_plot.reshape((N,N))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(xx, yy, z_plot, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()