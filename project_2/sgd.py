import functions
import numpy as np

class StocasticGradientDescent():
    def __init__(self, X_train, z_train, X_test, z_test):
        self.X_train = X_train
        self.z_train = z_train
        self.X_test = X_test
        self.z_test = z_test
        self.N = X_train.shape[0] # number of samples

    def gradient(self, X, z, theta, size, lmd=0):
        """
        Calculates gradient. If lmb == 0 we get OLS.
        For lmb != 0 we get ridge.
        """
        g = (2 / size) * X.T @ (X @ theta - z) + 2 * lmd * theta
        return g

    
    def learning_schedule(self, t, t0=5, t1=50):
        return t0 / (t + t1)

    def SGD(self, M, epochs, lmd=0, gamma=0.7, theta=None, fixed_eta=None):
        """
        SGD from week 40 slides
        
        -INPUT-
            M - Size of a batch
            epochs - Number of epochs
            theta - Initial guess (random if None)
            lmb - Regularization 
            gamma - Momentum
        """
        n_batch = int(self.N / M) # number of minibatches

        # Random theta if no initial guess is given
        if not theta:
            theta = np.random.randn(self.X_train.shape[1], 1)

        v = 0
        self.mse_values = np.zeros(epochs)
        np.random.seed(1337)
        j = np.arange(self.N)
        for epoch in range(epochs):
            np.random.shuffle(j)
            Xi = np.array_split(self.X_train[j], n_batch)
            zi = np.array_split(self.z_train[j], n_batch)
            for i in range(n_batch):
                random_index = np.random.randint(0, n_batch)
                gradient = self.gradient(Xi[random_index], zi[random_index], theta, M, lmd)
                if fixed_eta:
                    eta = fixed_eta
                else:
                    eta = self.learning_schedule(epoch*self.N) # t0=5 t1=50
                v = gamma * v + eta * gradient
                theta = theta - v

            # mse values
            z_pred = self.X_train @ theta
            self.mse_values[epoch-1] = functions.MSE(self.z_train, z_pred)
        
        return theta