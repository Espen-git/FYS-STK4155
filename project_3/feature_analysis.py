from load_data import load_data
from data_analysis import prep_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector

def feature_importance_from_coefficients(X, y):
    """
        Computes feature imprttance using the coefficients from SGD regression.

    input:
        - X
            Feature values / Data matrix.
        - y 
            Labels.
    output:
        - importance
            Values corresponding to the apparent importance of the different features.
    """
    sgd = SGDRegressor().fit(X, y)
    importance = np.abs(sgd.coef_)
    for _ in range(99):
        sgd = SGDRegressor().fit(X, y)
        importance += np.abs(sgd.coef_)

    importance = importance / 10.0 # average of the 10 runs
    return importance

def sequential_feature_selection(X_train, X_test, y_train, y_test):
    """
    Selects features using forward and backward sequential feature selection.
    Then plots for all possible number of features (1 - num_features).

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
    forward_scores = []
    backward_scores = []
    
    lasso = Lasso()
    for n in range(1, X_train.shape[1]):
        sfs_forward = SequentialFeatureSelector(lasso, n_features_to_select=n, direction="forward")
        sfs_backward = SequentialFeatureSelector(lasso, n_features_to_select=n, direction="backward")
    
        sfs_forward.fit(X_train, y_train)
        sfs_backward.fit(X_train, y_train)

        X_f = sfs_forward.transform(X_train)
        X_b = sfs_backward.transform(X_train)
        forward_model = SGDRegressor().fit(X_f, y_train)
        backward_model = SGDRegressor().fit(X_b, y_train)

        forward_scores.append(forward_model.score(sfs_forward.transform(X_test), y_test))
        backward_scores.append(backward_model.score(sfs_backward.transform(X_test), y_test))
    
    # Add for all features
    forward_scores.append(SGDRegressor().fit(X_train, y_train).score(X_test, y_test))
    backward_scores.append(SGDRegressor().fit(X_train, y_train).score(X_test, y_test))

    x = [1,2,3,4,5,6,7,8,9,10,11]
    plt.plot(x, forward_scores)
    plt.plot(x, backward_scores)
    plt.title("Forward v Backward scores")
    plt.xlabel("Number of features")
    plt.ylabel("R2 score")
    plt.legend(["Forward","Backward"])
    plt.show()

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

    if False: # Cange to True if you want to run this
        scaler = StandardScaler()
        scaler.fit(X_w)
        X_w_scaled = scaler.transform(X_w)

        scaler = StandardScaler()
        scaler.fit(X_r)
        X_r_scaled = scaler.transform(X_r)

        importance_w = feature_importance_from_coefficients(X_w_scaled, y_w)
        importance_r = feature_importance_from_coefficients(X_r_scaled, y_r)
        
        feature_names = np.array(X_w.columns)
        plt.bar(height=importance_w, x=feature_names)
        plt.title("White wine feature importance via coefficients")
        plt.show()
        feature_names = np.array(X_r.columns)
        plt.bar(height=importance_r, x=feature_names)
        plt.title("Red wine feature importance via coefficients")
        plt.show()

    if False: # Cange to True if you want to run this
        sequential_feature_selection(X_train_scaled_w, X_test_scaled_w, y_train_w, y_test_w)
        sequential_feature_selection(X_train_scaled_r, X_test_scaled_r, y_train_r, y_test_r)