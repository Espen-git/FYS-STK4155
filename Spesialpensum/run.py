from load_data import load_data
from TrainML import run_all_ML_models
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

features, labels, le = load_data()

# Correlation matrix
features_pd = pd.DataFrame(features)
corr_mat = features_pd.corr()
sb.heatmap(corr_mat, annot=True)
plt.show()

# min max avg of all features
df = pd.DataFrame(columns = ['min', 'max', 'mean'], index = features_pd.columns)
for feature in features_pd.columns:
    samples = features_pd[feature] # All samples of a feature/column
    df.loc[feature] = [np.min(samples), np.max(samples), np.mean(samples)]
pd.set_option('display.max_rows', len(df))
print(df)
pd.reset_option('display.max_rows')

# Plotting the data
sl = features[:,0].reshape((150,1))
sw = features[:,1].reshape((150,1))
pl = features[:,2].reshape((150,1))
pw = features[:,3].reshape((150,1))

sl_sw = np.concatenate((sl,sw), axis=1)
sl_pl = np.concatenate((sl,pl), axis=1)
sl_pw = np.concatenate((sl,pw), axis=1)
sw_pl = np.concatenate((sw,pl), axis=1)
sw_pw = np.concatenate((sw,pw), axis=1)
pl_pw = np.concatenate((pl,pw), axis=1)
all_combinations = np.array((sl_sw, sl_pl, sl_pw, sw_pl, sw_pw, pl_pw))

axis_labels = [['Speal Length', 'Sepal Width'], ['Speal Length', 'Petal Length'], ['Speal Length', 'Petal Width'], 
               ['Speal Width', 'Petal Length'], ['Speal Width', 'Petal Width'], ['Petal Lenght', 'Petal Width']]
for index, combination in enumerate(all_combinations):
    plt.scatter(combination[0:50,0], combination[0:50,1])
    plt.scatter(combination[50:100,0], combination[50:100,1])
    plt.scatter(combination[100:150,0], combination[100:150,1])
    plt.legend(['setosa', 'versicolor', 'virginica'])
    plt.xlabel(axis_labels[index][0])
    plt.ylabel(axis_labels[index][1])
    plt.show()

#run_all_ML_models()