from load_data import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def run_all_ML_models(X_train, X_test, y_train, y_test):
    """
    Trains and predicts using all three ML models.

    input:
        - X_train
            Traaining data
        - X_test
            Testing data
        - y_train 
            Training labels.
        - y_test
            Testing labels.
    output:
        - None
    """
    neigh = KNeighborsClassifier(n_neighbors=13).fit(X_train, y_train)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1918918918918919).fit(X_train)
    svc = SVC(kernel='poly', degree=1, gamma=0.12396396396396396).fit(X_train, y_train)
    
    neigh__test_score = neigh.score(X_test, y_test)
    neigh__train_score = neigh.score(X_train, y_train)
    kde_test_score = np.mean(kde.score_samples(X_test))
    kde_train_score = np.mean(kde.score_samples(X_train))
    svc_test_score = svc.score(X_test, y_test)
    svc_train_score = svc.score(X_train, y_train)
    print("KNN:")
    print(f"Test score: {neigh__test_score}")
    print(f"Train score: {neigh__train_score}")
    print("KD:")
    print(f"Test score: {kde_test_score}")
    print(f"Train score: {kde_train_score}")
    print("SVC:")
    print(f"Test score: {svc_test_score}")
    print(f"Train score: {svc_train_score}")
    

if __name__ == "__main__":
    features, labels, le = load_data()

    r = 2022 # random state seed
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=r)

    # scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    run_all_ML_models(X_train_scaled, X_test_scaled, y_train, y_test)