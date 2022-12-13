#import pandas as pd
import numpy as np
from sklearn import preprocessing

#import seaborn as sb

def load_data():
    """
    Reads data from file and returns numpy arrays with features and one-hot encoded labels.
    """
    data_file = open('data\iris.data', 'r')
    data = data_file.readlines() # Split lines
    

    features = np.empty((150, 4))
    labels_list = []
    for index, line in enumerate(data):
        split_line = line.split(",")
        features[index] = split_line[0:4]
        labels_list.append(split_line[4])

    le = preprocessing.LabelEncoder()
    le.fit(labels_list)
    labels = np.array(le.transform(labels_list))

    return features, labels, le