from load_data import load_data
from data_analysis import prep_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def scores(y_test, y_pred):
    """
    Prints MSE, R2 and MAD error scores for the predictions.

    input:
        - y_test
            True values.
        - y_pred
            predited values.
    output:
        - None
    """
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R2: {r2_score(y_test, y_pred)}")
    print(f"MAD: {mean_absolute_error(y_test, y_pred)}")

def accuracy(y_test, y_pred):
    """
    Rounds the prediction to neerest integer, then calculates accuracy.

    input:
        - y_test
            True values.
        - y_pred
            predited values.
    output:
        - None
    """
    y_pred = np.round(y_pred)
    res = y_pred==y_test
    print(np.sum(res) / len(y_test))

def run_all_ML_models(X_train, X_test, y_train, y_test):
    """
    Trains and predicts using all three ML models.

    input:
        - X_train
            Traaining data
        - X_test
            Testing data
        - y_train 
            Training labels.
        - y_test
            Testing labels.
    output:
        - None
    """
    lr = LinearRegression().fit(X_train, y_train)
    sgd = SGDRegressor(max_iter=1000, tol=1e-3, alpha=0.0001).fit(X_train, y_train)
    mlp = MLPRegressor(random_state=r, max_iter=1000).fit(X_train, y_train)
    #hidden_layer_sizes
    #activation
    #solver
    #alpha=0.0001
    #batch_size
    #learning_rate_init=0.001
    
    lr_y_pred = lr.predict(X_test)
    sgd_y_pred = sgd.predict(X_test)
    mlp_y_pred = mlp.predict(X_test)
    print("LR:")
    scores(y_test, lr_y_pred)
    print("SGD:")
    scores(y_test, sgd_y_pred)
    print("MLP:")
    scores(y_test, mlp_y_pred)

if __name__ == "__main__":
    white, red = load_data()
    X_w, y_w = prep_data(white)
    X_r, y_r = prep_data(red)

    r = 30 # random state seed
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w, y_w, test_size=0.3, random_state=r)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.3, random_state=r)

    # scaling
    scaler = StandardScaler()
    scaler.fit(X_train_w)
    X_train_scaled_w = scaler.transform(X_train_w)
    X_test_scaled_w = scaler.transform(X_test_w)
    scaler = StandardScaler()
    scaler.fit(X_train_r)
    X_train_scaled_r = scaler.transform(X_train_r)
    X_test_scaled_r = scaler.transform(X_test_r)

    run_all_ML_models(X_train_scaled_w, X_test_scaled_w, y_train_w, y_test_w)
    run_all_ML_models(X_train_scaled_r, X_test_scaled_r, y_train_r, y_test_r)