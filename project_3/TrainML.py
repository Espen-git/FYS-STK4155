from load_data import load_data
from data_analysis import prep_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def scores(y_test, y_pred):
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R2: {r2_score(y_test, y_pred)}")
    print(f"MAD: {mean_absolute_error(y_test, y_pred)}")

def accuracy(y_test, y_pred):
    y_pred = np.round(y_pred)
    res = y_pred==y_test
    print(np.sum(res) / len(y_test))

train = False
if train:
    r = 30 # random state seed
    X_train, X_test, y_train, y_test = train_test_split(X_w, y_w, test_size=0.3, random_state=r)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPRegressor(random_state=r, max_iter=500).fit(X_train_scaled, y_train)   
    #lr = LinearRegression().fit(X_train_scaled, y_train)
    #sgd = SGDRegressor(max_iter=1000, tol=1e-3).fit(X_train_scaled, y_train)
    mlp_y_pred = mlp.predict(X_test_scaled)
    #lr_y_pred = lr.predict(X_test_scaled)
    #sgd_y_pred = sgd.predict(X_test_scaled)
    print("MLP:")
    accuracy(y_test, mlp_y_pred)
    #scores(y_test, mlp_y_pred)
    """
    SGD og LR dåligere og ca. like
    print("LR:")
    scores(y_test, lr_y_pred)
    print("SGD:")
    scores(y_test, sgd_y_pred)
    """