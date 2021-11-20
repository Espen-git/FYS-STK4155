import functions
import analysis_functions
import cost_functions
import activation_functions
import NN as nn
import numpy as np
import matplotlib.pyplot as plt
from sgd import StocasticGradientDescent
import seaborn as sb
from numpy.core.numeric import NaN

def plot(MSE, x_values, y_values, x_labels, y_labels, title, filename=None):
    h = sb.heatmap(MSE, annot=True, fmt='.4g', cmap='YlGnBu',
                     xticklabels=x_values, yticklabels=y_values,
                     cbar_kws={'label': 'MSE'})
    h.set_xlabel(x_labels, size=12)
    h.set_ylabel(y_labels, size=12)
    h.invert_yaxis()
    h.set_title(title, size=15)
    if filename:
        plt.savefig('images/' + filename)

np.random.seed(1337)

N = 20 # n_samples
e = 0.1 # noise weight 
n = 5 # polynomial degree
x, y, z = functions.create_data(N, e)
X = functions.create_X(x, y, n)
X_train, X_test, z_train, z_test = analysis_functions.make_franke_data_ready_for_NN(x, y, z)

mse_ols = analysis_functions.ols(x, y, z)
print("MSE OLS: ", mse_ols)

M = 32
epochs = 300
eta = 1e-3
hidden = [50]
#analysis_functions.test_activation(X_train, X_test, z_train, z_test, M, epochs, eta, hidden, 'images/one_hidden_50')

hidden = [20,20]
#analysis_functions.test_activation(X_train, X_test, z_train, z_test, M, epochs, eta, hidden, 'images/two_hidden_20')

hidden = [10,10,10]
#analysis_functions.test_activation(X_train, X_test, z_train, z_test, M, epochs, eta, hidden, 'images/three_hidden_10')

def epochs_v_bsize(a_function_hidden, filename):
    n_in = X_train.shape[1]
    n_out = z_train.shape[1]
    hidden = [50]
    a_function_out = activation_functions.Identity()
    eta = 1e-3

    my_cost = cost_functions.MSE()
    epochs = range(50,501, 50)
    M_values = [8,16,32,64,128]

    MSE_values = np.zeros((len(M_values), len(epochs)))

    if isinstance(a_function_hidden, activation_functions.Sigmoid):
        title = "MSE of Sigmoid + Identity, epochs vs batch size"
    elif isinstance(a_function_hidden, activation_functions.ReLU):
        title = "MSE of ReLU + Identity, epochs vs batch size"
    elif isinstance(a_function_hidden, activation_functions.LeakyReLU):
        title = "MSE of Leaky ReLU + Identity, epochs vs batch size"

    print('Testing epochs v batch size for NN with franke data')
    plt.figure(figsize = [12, 8])
    for i in range(len(epochs)):
        print(i+1, '/', len(epochs))
        for j in range(len(M_values)):
            NN = nn.NN(n_in, n_out, a_function_hidden, a_function_out, my_cost,
                                    num_nodes_hidden_layers=hidden)
            layers = NN.make_layers()
            mse_train, mse_test = analysis_functions.train_NN_regression(NN, M_values[j], 
                                        epochs[i], eta, X_train, X_test, z_train, z_test)
            
            if mse_test[-1] < 1:
                MSE_values[j,i] = mse_test[-1]
            else:
                MSE_values[j,i] = NaN

    plot(MSE_values, epochs, M_values, "Epochs", 'Size of batches', title, filename=filename)
    plt.show()

def bsize_v_hidden(a_function_hidden, filename):
    n_in = X_train.shape[1]
    n_out = z_train.shape[1]
    a_function_out = activation_functions.Identity()
    eta = 1e-3
    my_cost = cost_functions.MSE()
    epochs = 300

    if isinstance(a_function_hidden, activation_functions.Sigmoid):
        title = "MSE of Sigmoid + Identity, hidden layers vs batch size"
    elif isinstance(a_function_hidden, activation_functions.ReLU):
        title = "MSE of ReLU + Identity, hidden layers vs batch size"
    elif isinstance(a_function_hidden, activation_functions.LeakyReLU):
        title = "MSE of Leaky ReLU + Identity, hidden layers vs batch size"

    n_hiddens = [[10], [50], [100], [10,10], [30,30], [50,50], [10,10,10], [15,15,15], [20,20,20]]
    M_values = [8, 16, 32, 64, 128]
    MSEs = np.zeros((len(n_hiddens), len(M_values)))
    plt.figure(figsize = [12, 8])

    print('Testing diffrent hidden layer stups v batch size for NN with franke data')
    for i in range(len(n_hiddens)):
        print(i+1, '/', len(n_hiddens))
        for j in range(len(M_values)):
            NN = nn.NN(n_in, n_out, a_function_hidden, a_function_out, my_cost,
                                    num_nodes_hidden_layers=n_hiddens[i])
            layers = NN.make_layers()
            mse_train, mse_test = analysis_functions.train_NN_regression(NN, M_values[j], epochs, eta,
                                                    X_train, X_test, z_train, z_test)
            if mse_test[-1] < 1:
                MSEs[i,j] = mse_test[-1]
            else: 
                MSEs[i,j] = NaN 


    plot(MSEs, M_values, n_hiddens, 'Size of batches', 'num_hidden', title, filename=filename)
    plt.show()

def eta_v_lmd(a_function_hidden, filename):
    n_in = X_train.shape[1]
    n_out = z_train.shape[1]
    n_hidden = [50]
    act_out = activation_functions.Identity()
    epochs = 300
    bsize = 16

    lmds = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    etas = [1e-4, 1e-3, 1e-2, 0.02, 0.03]

    my_cost = cost_functions.MSE()

    MSEs = np.zeros((len(etas), len(lmds)))

    if isinstance(a_function_hidden, activation_functions.Sigmoid):
        title = "MSE of Sigmoid + Identity, eta vs lmd"
    elif isinstance(a_function_hidden, activation_functions.ReLU):
        title = "MSE of ReLU + Identity, eta vs lmd"
    elif isinstance(a_function_hidden, activation_functions.LeakyReLU):
        title = "MSE of Leaky ReLU + Identity, eta vs lmd"

    print('Testing eta v lambda for NN with franke data')
    plt.figure(figsize = [12, 8])
    for i in range(len(etas)):
        print(i+1, '/', len(etas))
        for j in range(len(lmds)):
            NN = nn.NN(n_in, n_out, a_function_hidden, act_out, my_cost,
                                    num_nodes_hidden_layers=n_hidden)
            layers = NN.make_layers()
            mse_train, mse_test = analysis_functions.train_NN_regression(NN, bsize, epochs, etas[i],
                                                    X_train, X_test, z_train, z_test, lmd=lmds[j])
            if mse_test[-1] < 1:
                MSEs[i,j] = mse_test[-1]
            else: 
                MSEs[i,j] = NaN 

    plot(MSEs, lmds, etas, 'lambdas', 'Learning rate', title, filename=filename)
    plt.show()

# Epochs v batch size 
#epochs_v_bsize(activation_functions.Sigmoid(), 'epoch_v_batch_sigmoid')
#epochs_v_bsize(activation_functions.ReLU(), 'epoch_v_batch_relu')
#epochs_v_bsize(activation_functions.LeakyReLU(), 'epoch_v_batch_lrelu')

# Hidden layers v batch size 
#bsize_v_hidden(activation_functions.Sigmoid(), 'layer_v_batch_sigmoid')
#bsize_v_hidden(activation_functions.ReLU(), 'layer_v_batch_relu')
#bsize_v_hidden(activation_functions.LeakyReLU(), 'layer_v_batch_lrelu')

# eta v lambda 
#eta_v_lmd(activation_functions.Sigmoid(), 'eta_v_lmd_sigmoid')
#eta_v_lmd(activation_functions.ReLU(), 'eta_v_lmd_relu')
#eta_v_lmd(activation_functions.LeakyReLU(), 'eta_v_lmd_lrelu')