import functions

class StocasticGradientDescent():
    def __init__(self, X_train, z_train, X_test, z_test):
        self.X_train = X_train
        self.z_train = z_train
        self.X_test = X_test
        self.z_test = z_test
        self.N = X_train.shape[0] # number of samples

    def gradient(self, X, z, theta, size, lmb=0):
        """
        Calculates gradient. If lmb == 0 we get OLS.
        For lmb != 0 we get ridge.
        """
        g = (2 / size) * X.T @ (X @ theta - z) + 2*lmb * theta
        return g

    
    def learning_schedule(self, t0, t1, t)
        return t0 / (t + t1)

    def SGD(self, M, epochs, theta=None, lmb=0, t0=5, t1=50):
        """
        SGD from week 40 slides
        
        -INPUT-
            M - Size of a batch
            epochs - Number of epochs
            theta - Initial guess
            lmb - Outlier penalization fro ridge
            t0, t1 - Values controling decay of learning rate
        """
        n_batch = int(self.N / M) # number of minibatches

        # Random theta if no initial guess is given
        if not theta:
            theta = np.random.randn(self.X_train.shape[1], 1)

        for epoch in range(1, epochs+1):
            for i in range(n_batch):
                random_index = M * np.random.randint(n_batch)
                Xi = X[random_index:random_index+M] 
                zi = z[random_index:random_index+M]
                gradients = self.gradient(Xi, zi, theta, M, lmb)
                eta = learning_schedule(t0, t1, epoch*n_batch+i)
                theta = theta - eta*gradients
        
            # mse values
            self.mse_values = np.zeros(epochs)
            z_tilde = self.X_test @ theta
            self.mse_values[epoch-1] = functions.MSE(self.z_test, z_tilde)
        
        return theta