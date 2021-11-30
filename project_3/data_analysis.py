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

def scores(y_test, y_pred):
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R2: {r2_score(y_test, y_pred)}")
    print(f"MAD: {mean_absolute_error(y_test, y_pred)}")

def accuracy(y_test, y_pred):
    y_pred = np.round(y_pred)
    res = y_pred==y_test
    print(np.sum(res) / len(y_test))


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

#print("Red wine")
#feature_table(X_r)
#print("-"*50)
#print("White wine")
#feature_table(X_w)

#target_histogram(y_r, 'Red wine')
#target_histogram(y_w, 'White wine')

#cov = X_w.cov()
#val, vec = np.linalg.eig(cov)
# 4 3 5 6 2
#res = np.zeros(vec.shape)
#for i in range(len(val)):
#    res[i] = vec[i] * val[i]
#res = np.sum(res, axis=0)
#print(res) # 4 3 8 (6 5)  
#print(col_names)

# remove free sulphur and density and alcohol
#features_w = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'pH', 'sulphates']
#selected_w = X_w[features_w]
#print(selected_w)
#X_w = selected_w

train = True
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