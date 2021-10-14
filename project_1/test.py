import numpy as np
#import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split
from ex1 import OLS, create_X
from franke import FrankeFunction
from ex1 import OLS, MSE, R2, create_X
from franke import FrankeFunction
from random import seed
from random import randrange
 
# Split a dataset into k folds
def cross_validation_split(dataset, folds=10):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Make data set
n = 40
x,y = xy_data_2(n)
z = FrankeFunction_noisy(x,y)

# number of k-folds
k_folds = 10

# Polynomial fit
polynomial = 10

# Stacking x and y 
x_and_y=np.hstack((x.ravel().reshape(x.ravel().shape[0],1),y.ravel().reshape(y.ravel().shape[0],1)))

# Scaling data
scaler = StandardScaler()
scaler.fit(x_and_y)
x_and_y_scaled = scaler.transform(x_and_y)

# Make list and arrays to store results
all_r2_ols_cv=[]
mean_r2_ols_cv=[]
error = np.zeros(polynomial)
bias = np.zeros(polynomial)
variance = np.zeros(polynomial)
polydegree = np.zeros(polynomial)
train_error = np.zeros(polynomial)


for poly in range(polynomial):
    # Make list and arrays to store results
    r2_ = []
    
    #Make array to store predictions
    pred_test = np.empty((int(z.ravel().shape[0]*(1/k_folds)), k_folds))
    pred_train = np.empty((int(z.ravel().shape[0]*(1-(1/k_folds))), k_folds))
    
    # Stacking x , y (X) and z 
    data = np.hstack((x_and_y_scaled,z.ravel().reshape(n**2,1)))
    
    #Make folds 
    folds = cross_validation_split(data, k_folds)
    for i in range(k_folds):
        #Make train and test data using the i'th fold
        n_fold = folds.copy()
        test_data = n_fold.pop(i)
        test_data= np.asarray(test_data)
        train_data = np.vstack(n_fold)
        
        #split z and X
        z_train = train_data[:,-1]
        xy_train = train_data[:,0:-1]
        z_test = test_data[:,-1]
        xy_test = test_data[:,0:-1]
        
        # Fit training data
        X_train = make_X_matrix(xy_train.T[0],xy_train.T[1],poly+1)
        beta = calc_beta(X_train, z_train)
        
        # Do prediction on test and train data
        z_pred_test=predict(xy_test.T[0],xy_test.T[1],poly+1,beta)
        z_pred_train=predict(xy_train.T[0],xy_train.T[1],poly+1,beta)
        pred_test[:,i]=predict(xy_test.T[0],xy_test.T[1],poly+1,beta)
        pred_train[:,i]=predict(xy_train.T[0],xy_train.T[1],poly+1,beta)
        
        # Append results to arrays and lists
        r2_.append(r2_score(z_test,z_pred_test))
        
    train_error[poly] = np.mean( np.mean((z_train.reshape(z_train.shape[0],1) - pred_train)**2, axis=1, keepdims=True) )   
    error[poly] = np.mean( np.mean((z_test.reshape(z_test.shape[0],1) - pred_test)**2, axis=1, keepdims=True) )
    bias[poly] = np.mean( (z_test.reshape(z_test.shape[0],1) - np.mean(pred_test, axis=1, keepdims=True))**2 )
    variance[poly] = np.mean( np.var(pred_test, axis=1, keepdims=True) )

    print('Polynomial degree:', poly+1)
    print('Error:', error[poly])
    print('Bias^2:', bias[poly])
    print('Var:', variance[poly])
    print('{} >= {} + {} = {}'.format(error[poly], bias[poly], variance[poly], bias[poly]+variance[poly]))
        
    #plotting prediction based on all data     
    z_pred_for_plot = predict(x_and_y_scaled.T[0],x_and_y_scaled.T[1],poly+1,beta)
    fig = plt.figure(figsize=(32,12))
    ax = fig.gca(projection ='3d')
    surf = ax.plot_surface(x,y,z_pred_for_plot.reshape(n,n),cmap=cm.coolwarm, linewidth = 0, antialiased=False)
    ax.set_zlim(-0.10,1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf,shrink=0.5, aspect=5)
    fig.suptitle("A {} degree polynomial fit of Franke function using OLS and K-fold crossval".format(poly+1) ,fontsize="40", color = "black")
    fig.show()
        
    all_r2_ols_cv.append(r2_)
    mean_r2_ols_cv.append(np.mean(r2_))