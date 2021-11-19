import functions
from sgd import StocasticGradientDescent
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

def plot(MSE, x, y, x_label, y_label, title, filename=None):
    h = sb.heatmap(MSE, annot=True, fmt='.4g', 
                cmap='YlGnBu', xticklabels=x, 
                yticklabels=y, cbar_kws={'label':'MSE'})
    h.set_xlabel(x_label, size=12)
    h.set_ylabel(y_label, size=12)
    h.invert_yaxis()
    h.set_title(title, size=15)
    if filename:
        plt.savefig('images/' + filename)

def ols_epoch_v_batch(clf):
    eta = 0.01
    gamma = 0.6

    M_values = [4, 8, 16, 32, 64] # bach sizes
    epoch_values = [500, 750, 1000, 1250, 1500, 1750, 2000]

    MSE_values = np.zeros((len(epoch_values), len(M_values)))
    print("Epoch v batch, OLS")
    for i in range(len(epoch_values)):
        print(f"Epoch: {i+1} / {len(epoch_values)}")
        for j in range(len(M_values)):
            theta = clf.SGD(M_values[j], epoch_values[i], gamma=gamma)
            MSE_values[i, j] = clf.mse_values[-1]

    plot(MSE_values, M_values, epoch_values, 'Minibatch size', 'Epochs',
             'MSE, OLS epoch v batch', 'ols_epoch_batch')
    plt.show()

def ols_eta_v_batch(clf):
    gamma = 0.6
    epochs = 1500

    eta_values = [0.0001, 0.001, 0.01, 0.02]
    M_values = [4, 8, 16, 32] # bach sizes

    MSE_values = np.zeros((len(eta_values), len(M_values)))
    print("eta v batch, OLS")
    for i in range(len(eta_values)):
        print(f"eta: {i+1} / {len(eta_values)}")
        for j in range(len(M_values)):
            theta = clf.SGD(M_values[j], epochs, fixed_eta=eta_values[i],
                     gamma=gamma)
            MSE_values[i, j] = clf.mse_values[-1]

    plot(MSE_values, M_values, eta_values, 'Minibatch size', 'eta',
             'MSE, OLS eta v batch', 'ols_eta_batch')
    plt.show()

def ols_eta_v_gamma(clf):
    epochs = 1500
    M = 8

    eta_values = [0.01, 0.025, 0.05]
    gamma_values = [0.4, 0.5, 0.6, 0.7]
    
    MSE_values = np.zeros((len(eta_values), len(gamma_values)))
    for i in range(len(eta_values)):
        print(f"eta: {i+1} / {len(eta_values)}")
        for j in range(len(gamma_values)):
            theta = clf.SGD(M, epochs, fixed_eta=eta_values[i],
                     gamma=gamma_values[j])
            MSE_values[i, j] = clf.mse_values[-1]
    
    plot(MSE_values, gamma_values, eta_values, 'gamma', 'eta',
             'MSE, OLS eta v gamma', 'ols_eta_gamma')
    plt.show()


"""
def ridge_epoch_v_batch():
def ridge_eta_v_lmb():
def ridge_gamma_v_lmb():
def ridge_eta_v_gamma():
def ridge_gamma_v_lambda_best():
def ridge_t_best():
"""
###def ridge_eta_v_batch():
###def ridge_eta_v_gamma():

N = 20 # n_samples
e = 0.1 # noise weight 
n = 5 # polynomial degree
x,y,z = functions.create_data(N, e)
X = functions.create_X(x, y, n)
X_train, X_test, z_train, z_test = functions.make_franke_data_ready_for_regression(X, z, n)

clf = StocasticGradientDescent(X_train, z_train, X_test, z_test)

# OLS testing
#ols_epoch_v_batch(clf)
#ols_eta_v_batch(clf)
ols_eta_v_gamma(clf)