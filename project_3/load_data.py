import pandas as pd
import numpy as np
import seaborn as sb

def load_data():
    """
    Reads data from csv file (is actualy ; seperated) and returns datarame objects with the data.

    input: 
        - None
    output:
        - white
            Pandas dataframe containing white wine sample data with features and labels.
        - red
            Pandas dataframe containing red wine sample data with features and labels.
    """
    # Loads red and white wine data from .csv files
    red = pd.read_csv(r'data\winequality-red.csv',sep=";")
    white = pd.read_csv(r'data\winequality-white.csv',sep=";")
    return white, red