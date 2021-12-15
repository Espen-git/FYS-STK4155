from load_data import load_data
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

def correlation_matrix(X, plot=True):
    """
    Creates the correlation matrix and plots it if plot=True.
    """
    corr_X = X.corr() # correlation matrix
    if plot:
        f, ax = plt.subplots()
        sb.heatmap(abs(corr_X), mask=np.zeros_like(abs(corr_X), dtype=np.bool), cmap=sb.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
        plt.show()

def prep_data(data):
    col_names = data.columns # names of the features/columns
    X = data[col_names[:-1]] # remove targets
    y = data[col_names[-1]] # only targets
    return X, y

def feature_table(X):
    df = pd.DataFrame(columns = ['min', 'max', 'mean'], index = X.columns)
    for feature in X.columns:
        samples = X[feature] # All samples of a feature/column
        df.loc[feature] = [np.min(samples), np.max(samples), np.mean(samples)]
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

def target_histogram(y, title):
    bins = list(range(min(y), max(y)+2))
    hist = plt.hist(y, bins=bins, align='left')
    plt.xlabel('Sensory preference')
    plt.ylabel('Number of samples')
    plt.title(title)
    plt.show()

white, red = load_data()
X_w, y_w = prep_data(white)
X_r, y_r = prep_data(red)
col_names = X_w.columns

print("Red wine")
feature_table(X_r)
print("-"*50)
print("White wine")
feature_table(X_w)

target_histogram(y_r, 'Red wine')
target_histogram(y_w, 'White wine')

corr_w = X_w.corr()
sb.heatmap(corr_w)
plt.show()

corr_r = X_r.corr()
sb.heatmap(corr_r)
plt.show()