import functions
from sgd import StocasticGradientDescent
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

def plot_heatmap():

#def ols_epoch_v_batch():
#def ols_eta_v_gamma():
#def ols_eta_v_batch():

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
regression_data = functions.make_data_ready_for_regression(X, z, n)
X_train = regression_data['X_train_scaled']
z_train = regression_data['z_train_scaled']
X_test = regression_data['X_test_scaled']
z_test = regression_data['z_test_scaled']

clf = StocasticGradientDescent(X_train, z_train, X_test, z_test)

# OLS testing
#ols_epochs_batches()
#ols_eta_gamma()
#ols_eta_batches()