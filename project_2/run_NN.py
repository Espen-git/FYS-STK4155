import functions
import analasys_functions
import cost_functions
import activation_functions
import NN
import numpy as np
import matplotlib.pyplot as plt
from sgd import StocasticGradientDescent
import seaborn as sb

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
x,y,z = functions.create_data(N, e)
X = functions.create_X(x, y, n)
X_train, X_test, z_train, z_test = functions.make_franke_data_ready_for_NN(X, z, n)