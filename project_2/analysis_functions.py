import functions
import NN as nn
import numpy as np
import matplotlib.pyplot as plt
import activation_functions
import cost_functions
import pandas as pd
from sklearn.datasets import load_breast_cancer

def make_franke_data_ready_for_NN(x, y, z):
    inputs = np.stack((x, y), axis=1)
    z = z.reshape(z.size, 1)
    X_train, X_test, z_train, z_test = functions.make_franke_data_ready_for_regression(inputs, z)

    return X_train, X_test, z_train, z_test

def make_dc_data_ready_for_NN():
    cancer_data = load_breast_cancer()

    inputs = cancer_data.data # Feature matrix of 569 rows (samples) and 30 columns (features)
    outputs = cancer_data.target # array of 569 rows, 0 for benign and 1 for malignant
    outputs = outputs.reshape(outputs.shape[0], 1)
    
    X_train, X_test, z_train, z_test = functions.make_franke_data_ready_for_regression(inputs, outputs)
    z_train = np.where(z_train > 0.5, 1, 0)
    z_test = np.where(z_test > 0.5, 1, 0)

    return X_train, X_test, z_train, z_test

def ols(x, y, z):
    n = 5
    X = functions.create_X(x, y, n)
    X_train, X_test, z_train, z_test = functions.make_franke_data_ready_for_regression(X, z)

    # Finds beat beta and MSE value
    beta = np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ z_train)
    z_pred_ols = X_test @ beta
    mse_ols = functions.MSE(z_pred_ols, z_test)

    return mse_ols

def plot(values, label, x_labels, y_labels, title, filename=None):
    plt.plot(values, label = label)
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.title(title)
    plt.legend()

    if filename:
        plt.savefig(filename) 

def train_NN_classifier(NN, M, epochs, eta, X_train, X_test, z_train, z_test):
    batches = int(z_train.shape[0] / M)

    training_accuracy = []
    test_accuracy = []
    # Train network
    index = np.arange(len(X_train))
    for _ in range(epochs):
        np.random.shuffle(index)
        X_batches = np.array_split(X_train[index], batches)
        z_batches = np.array_split(z_train[index], batches)

        for l in range(batches):
            r = np.random.randint(0, batches-1)
            if NN.num_nodes_each_h:
                NN.back_prop(X_batches[r], z_batches[r], eta=eta)
            else:
                NN.logistic(X_batches[r], z_batches[r], eta=eta)

        z_pred_train = NN.forward(X_train)
        z_pred_test = NN.forward(X_test)
        z_pred_test = np.where(z_pred_test > 0.5, 1, 0)
        z_pred_train = np.where(z_pred_train > 0.5, 1, 0)

        training_accuracy.append(functions.Accuracy(z_pred_train, z_train))
        test_accuracy.append(functions.Accuracy(z_pred_test, z_test))

    return training_accuracy, test_accuracy

def train_NN_regression(NN, M, epochs, eta, X_train, X_test, z_train, z_test, lmd=0):
    batches = int(z_train.shape[0] / M)
    mse_values_test = []
    mse_values_training = []
    # train network
    index = np.arange(len(X_train))
    for k in range(epochs):
        np.random.shuffle(index)
        X_batches = np.array_split(X_train[index], batches)
        z_batches = np.array_split(z_train[index], batches)

        for _ in range(batches):
            r = np.random.randint(0, batches)
            NN.back_prop(X_batches[r], z_batches[r], eta=eta, lmd=lmd)

        z_pred_train = NN.forward(X_train)
        z_pred_test = NN.forward(X_test)

        mse_train = functions.MSE(z_pred_train, z_train)
        mse_test = functions.MSE(z_pred_test, z_test)

        mse_values_training.append(mse_train)
        mse_values_test.append(mse_test)

    return mse_values_training, mse_values_test

def run_NN_classifier(X_train, X_test, z_train, z_test, M, epochs, eta, a_function_hidden, a_function_out, cost_function, hidden, lmd=0):
    num_input = X_train.shape[1]
    num_output = z_train.shape[1]

    NN = nn.NN(num_input, num_output, a_function_hidden, a_function_out, cost_function, num_nodes_hidden_layers=hidden)
    layers = NN.make_layers()
    accuracy_train, accuracy_test = train_NN_classifier(NN, M, epochs, eta, X_train, X_test, z_train, z_test, lmd=lmd)

    # Predictions from trained neural network
    z_pred_train = NN.feed_forward(X_train)
    z_pred_test = NN.feed_forward(X_test)

    pred_test = np.where(z_pred_test > 0.5, 1, 0)
    pred_train = np.where(z_pred_train > 0.5, 1, 0)

    return accuracy_train, accuracy_test

def test_activation(X_train, X_test, z_train, z_test, M, epochs, eta, hidden, filename):
    n_in = X_train.shape[1]
    n_out = z_train.shape[1]
    activation_out = activation_functions.Identity()
    cost_function = cost_functions.MSE()
    
    a_functions = [activation_functions.Sigmoid(), activation_functions.ReLU(), activation_functions.LeakyReLU()]

    # One layer
    for i in range(len(a_functions)):
        NN = nn.NN(n_in, n_out, a_functions[i], activation_out, cost_function, num_nodes_hidden_layers=hidden)
        layers = NN.make_layers()

        #Testing MSE before training the network
        z_pred = NN.forward(X_test)
        print('MSE before training (' + a_functions[i].name + ') : ', functions.MSE(z_pred, z_test))

        mse_train, mse_test = train_NN_regression(NN, M, epochs, eta, X_train, X_test,
                                                  z_train, z_test)

        # Trained NN on testing data
        z_pred_train = NN.forward(X_train)
        z_pred_test = NN.forward(X_test)

        print(f"MSE train ({a_functions[i].name}): {functions.MSE(z_pred_train, z_train)}")
        print(f"MSE test ({a_functions[i].name}): {functions.MSE(z_pred_test, z_test)}")
        plot(mse_test, a_functions[i].name, 'Epochs', 'MSE', 'Testing Activations functions',
             filename=filename)
        plt.show()

    return mse_train, mse_test