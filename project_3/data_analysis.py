from load_data import load_data
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

def correlation_matrix(X, plot=True):
    """
    Plots the correlation matrix of the input data X.

    input:
        - X
            Sample data with multiple features.
    output:
        - corr_X
            Correlation matrix of the data in X.
    """
    corr_X = X.corr() # correlation matrix
    if plot:
        f, ax = plt.subplots()
        sb.heatmap(abs(corr_X), mask=np.zeros_like(abs(corr_X)), cmap=sb.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
        plt.show()
    return corr_X

def prep_data(data):
    """
    Splits the data into X (samples v. features) and y (samples v. labels) so it can be used in ML.

    input:
        - data
            Pandas dataframe with features and labels combined.
    output:
        - X
            Dataframe containing only features.
        - y
            Dataframe containing only labels.
    """
    col_names = data.columns # names of the features/columns
    X = data[col_names[:-1]] # remove targets
    y = data[col_names[-1]] # only targets
    return X, y

def feature_table(X):
    """
    Prints a table of the features with their minimum, maximum and mean values.

    input:
        - X
            Dataframe of all the samples.
    output:
        - None
    """
    df = pd.DataFrame(columns = ['min', 'max', 'mean'], index = X.columns)
    for feature in X.columns:
        samples = X[feature] # All samples of a feature/column
        df.loc[feature] = [np.min(samples), np.max(samples), np.mean(samples)]
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

def target_histogram(y, title):
    """
    Plots the distribution of the target values.

    input:
        - y
            Dataframe/arry of labels (0-10).
        - title
            Plot title.
    output:
        - None
    """
    bins = list(range(min(y), max(y)+2))
    hist = plt.hist(y, bins=bins, align='left')
    plt.xlabel('Sensory preference')
    plt.ylabel('Number of samples')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
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

    corr_w = correlation_matrix(X_w)
    corr_r = correlation_matrix(X_r)