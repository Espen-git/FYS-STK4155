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
    gamma = 0.7

    M_values = [2, 4, 8, 16, 32, 64] # bach sizes
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
    gamma = 0.7
    epochs = 1500

    eta_values = [0.0001, 0.001, 0.01, 0.1]
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

    eta_values = [0.001, 0.01, 0.1]
    gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
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

def ridge_epoch_v_batch(clf):
    lmd = 1e-5
    eta = 0.01
    gamma = 0.7
    M_values = [4, 8, 16, 32]
    epochs = [500, 750, 1000, 1250, 1500, 1750, 2000]

    MSE_values = np.zeros((len(M_values), len(epochs)))
    for i in range(len(M_values)):
        print(f"Batch: {i+1} / {len(M_values)}")
        for j in range(len(epochs)):
            theta = clf.SGD(M_values[i], epochs[j], fixed_eta=eta, gamma=gamma, lmd=lmd)
            MSE_values[i, j] = clf.mse_values[-1]
    
    plot(MSE_values, epochs, M_values, 'epochs', 'batch',
             'MSE, Ridge epochs v batch', 'ridge_epoch_batch')
    plt.show()

def ridge_eta_v_batch(clf):
    lmd = 1e-5
    eta_values = [0.001, 0.01, 0.1]
    gamma = 0.7
    M_values = [4, 8, 16, 32]
    epochs = 1500

    MSE_values = np.zeros((len(M_values), len(eta_values)))
    for i in range(len(M_values)):
        print(f"Batch: {i+1} / {len(M_values)}")
        for j in range(len(eta_values)):
            theta = clf.SGD(M_values[i], epochs, fixed_eta=eta_values[j], gamma=gamma, lmd=lmd)
            MSE_values[i, j] = clf.mse_values[-1]
    
    plot(MSE_values, eta_values, M_values, 'eta', 'batch',
             'MSE, Ridge eta v batch', 'ridge_eta_batch')
    plt.show()

def ridge_eta_v_gamma(clf):
    lmd = 1e-5
    eta_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.02, 0.03]
    gamma_values = [0.5, 0.6, 0.7, 0.8]
    M = 8
    epochs = 1500

    MSE_values = np.zeros((len(gamma_values), len(eta_values)))
    for i in range(len(gamma_values)):
        print(f"Gamma: {i+1} / {len(gamma_values)}")
        for j in range(len(eta_values)):
            theta = clf.SGD(M, epochs, fixed_eta=eta_values[j], gamma=gamma_values[i], lmd=lmd)
            MSE_values[i, j] = clf.mse_values[-1]
    
    plot(MSE_values, eta_values, gamma_values, 'eta', 'gamma',
             'MSE, Ridge eta v gamma', 'ridge_eta_gamma')
    plt.show()

def ridge_eta_v_lmb(clf):
    lmd_values = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1]
    eta_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.02, 0.03]
    gamma= 0.7
    M = 8
    epochs = 1500

    MSE_values = np.zeros((len(lmd_values), len(eta_values)))
    for i in range(len(lmd_values)):
        print(f"Lambda: {i+1} / {len(lmd_values)}")
        for j in range(len(eta_values)):
            theta = clf.SGD(M, epochs, fixed_eta=eta_values[j], gamma=gamma, lmd=lmd_values[i])
            MSE_values[i, j] = clf.mse_values[-1]
    
    plot(MSE_values, eta_values, lmd_values, 'eta', 'lambda',
             'MSE, Ridge eta v lambda', 'ridge_eta_lmd')
    plt.show()

def ridge_gamma_v_lmb(clf):
    lmd_values = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1]
    eta = 0.01
    gamma_values = [0.5, 0.6, 0.7, 0.8]
    M = 8
    epochs = 1500

    MSE_values = np.zeros((len(lmd_values), len(gamma_values)))
    for i in range(len(lmd_values)):
        print(f"lambda: {i+1} / {len(lmd_values)}")
        for j in range(len(gamma_values)):
            theta = clf.SGD(M, epochs, fixed_eta=eta, gamma=gamma_values[j], lmd=lmd_values[i])
            MSE_values[i, j] = clf.mse_values[-1]
    
    plot(MSE_values, gamma_values, lmd_values, 'gamma', 'lambda',
             'MSE, Ridge gamma v lambda', 'ridge_gamma_lmd')
    plt.show()

def ols_best(clf):
    eta = 0.01
    gamma = 0.7
    M = 8
    epochs = 1500

    print("OLS Best:")
    theta = clf.SGD(M, epochs, gamma=gamma, fixed_eta=eta)
    z_pred = clf.X_test @ theta
    mse = functions.MSE(clf.z_test, z_pred)
    print(f"MSE: {mse}")

def ridge_best(clf):
    lmd = 0.01
    eta = 0.01
    gamma = 0.7
    M = 8
    epochs = 1500

    print("Best Ridge:")
    theta = clf.SGD(M, epochs, fixed_eta=eta, gamma=gamma, lmd=lmd)
    z_pred = clf.X_test @ theta
    mse = functions.MSE(clf.z_test, z_pred)
    print(f"MSE: {mse}")

N = 20 # n_samples
e = 0.1 # noise weight 
n = 5 # polynomial degree
x,y,z = functions.create_data(N, e)
X = functions.create_X(x, y, n)
X_train, X_test, z_train, z_test = functions.make_franke_data_ready_for_regression(X, z)

clf = StocasticGradientDescent(X_train, z_train, X_test, z_test)
# OLS paramater testing
#ols_epoch_v_batch(clf)
#ols_eta_v_batch(clf)
#ols_eta_v_gamma(clf)

# Rige paramater testing
#ridge_epoch_v_batch(clf)
#ridge_eta_v_batch(clf)
#ridge_eta_v_gamma(clf)
#ridge_eta_v_lmb(clf)
#ridge_gamma_v_lmb(clf)

# MSE for best paramaters using test data
ols_best(clf)
ridge_best(clf)