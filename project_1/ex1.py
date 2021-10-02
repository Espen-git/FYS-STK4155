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

# From week35 slides
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

if __name__ == "__main__":
    n = 5 # polynomial degree
    N = 100 # number of samples
    e = 0.1 # noise weight
    show_plot = False

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)

    z = FrankeFunction(xx, yy) # True values
    # add noise
    np.random.seed(2021)
    noise = np.random.randn(N,N) # NxN matrix of normaly distributed noise
    z = z + e*noise

    X = create_X(x, y, n) # Creates the design matrix
    X_train, X_test, z_train, z_test = train_test_split(X, z, 
        test_size=0.2, random_state=2021)

    """
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
    
    beta = OLS(X_train, z_train)
    ztilde = X_train @ beta
    zpredict = X_test @ beta

    # Compute confidence intervals for beta
    mean_beta = np.mean(beta, 1) # Mean value over beta_i
    std_beta = np.std(beta, 1) # Standard deviation over beta_i
    z_confidence = 1.96 # Confidence of 95%

    confidence = np.zeros([len(beta)])
    for i in range(len(beta)):
        confidence[i] = (z_confidence * std_beta[i]) / np.sqrt(N)

    if show_plot:
        plt.errorbar(range(len(beta)), mean_beta, yerr=confidence, fmt='.')
        plt.xlabel('beta_i')
        plt.ylabel('mean')
        plt.title('Confidence intervals of beta_i')
        plt.show()

    # MSE
    # Scikit-learns MSE function
    mse_train = mean_squared_error(z_train, ztilde)
    mse_test = mean_squared_error(z_test, zpredict)

    #R^2
    # Scikit-learns r2 function
    r2_train = r2_score(z_train, ztilde)
    r2_test = r2_score(z_test, zpredict)

    print(f"MSE train: {mse_train}")
    print(f"MSE test: {mse_test}")
    print(f"R2 train: {r2_train}")
    print(f"R2 test: {r2_test}")

    """
    if show_plot:
        # plot ztilde
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plot_x = np.linspace(0, 1, len(ztilde))
        plot_y = np.linspace(0, 1, len(ztilde))
        plot_xx, plot_yy = np.meshgrid(plot_x, plot_y)
        surf = ax.plot_surface(plot_xx, plot_yy, ztilde, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
    """