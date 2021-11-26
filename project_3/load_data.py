import pandas as pd
import numpy as np
import seaborn as sb

def load_data():
    # Loads red and white wine data from .csv files
    red = pd.read_csv(r'data\winequality-red.csv',sep=";")
    white = pd.read_csv(r'data\winequality-white.csv',sep=";")
    return white, red