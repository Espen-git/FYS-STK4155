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
 
for i in range(5):
    print('Test error, lambda = ' + str(i)) 