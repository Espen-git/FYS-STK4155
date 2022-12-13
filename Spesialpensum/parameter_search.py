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

def search_KD(X_train, X_test):
    kernel_list = ['gaussian', 'exponential']
    h_list = np.linspace(0.1, 1, 1000)
    test_scores = np.zeros((len(kernel_list), len(h_list)))
    train_scores = np.zeros((len(kernel_list), len(h_list)))
    best_score = -np.inf
    for index1,  kernel in enumerate(kernel_list):
        for index2, h in enumerate(h_list):
            kde = KernelDensity(kernel=kernel, bandwidth=h).fit(X_train)
            test_score = np.mean(kde.score_samples(X_test))
            train_score = np.mean(kde.score_samples(X_train))
            test_scores[index1,index2] = test_score
            train_scores[index1,index2] = train_score
            if (test_score >= best_score):
                best_score = test_score
                best_kernel = kernel
                best_h = h

    print(f'Best score: {best_score}. With {best_kernel} and h = {best_h}')

    plt.plot(h_list, test_scores[0,:])
    plt.plot(h_list, test_scores[1,:])
    plt.plot(h_list, train_scores[0,:])
    plt.plot(h_list, train_scores[1,:])
    plt.legend(['Gaussian test','Exponential test', 'Gaussian train','Exponential train'])
    plt.xlabel('h')
    plt.ylabel('Mean Log-likelihood')
    plt.show()

def search_KNN(X_train, X_test, y_train, y_test):
    k_list = np.arange(1,51)
    best_score = 0
    test_scores = np.zeros((len(k_list)))
    train_scores = np.zeros((len(k_list)))
    for index, k in enumerate(k_list):
        neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        test_score = neigh.score(X_test, y_test)
        train_score = neigh.score(X_train, y_train)
        test_scores[index] = test_score
        train_scores[index] = train_score
        if (test_score >= best_score):
            best_score = test_score
            best_k = k

    print(f'Best score: {best_score}. With k = {best_k}')

    plt.plot(k_list, test_scores)
    plt.plot(k_list, train_scores)
    plt.legend(['Test','Train'])
    plt.xlabel('k')
    plt.ylabel('Mean accuracy')
    plt.show()

def search_SVC(X_train, X_test, y_train, y_test):
    kernel = 'poly' # polynomial kernel
    deg_list = np.arange(1,6)
    gamma_list = np.linspace(0.01, 1, 1000)
    test_scores = np.zeros((len(deg_list), len(gamma_list)))
    train_scores = np.zeros((len(deg_list), len(gamma_list)))
    best_score = 0
    for index1,  deg in enumerate(deg_list):
        for index2, gamma in enumerate(gamma_list):
            svc = SVC(kernel=kernel, degree=deg, gamma=gamma).fit(X_train, y_train)
            test_score = svc.score(X_test, y_test)
            train_score = svc.score(X_train, y_train)
            test_scores[index1,index2] = test_score
            train_scores[index1,index2] = train_score
            if (test_score > best_score):
                best_score = test_score
                best_deg = deg
                best_gamma = gamma

    print(f'Best score: {best_score}. With deg = {best_deg} and gamma = {best_gamma}')

    plt.plot(gamma_list, test_scores[0,:])
    plt.plot(gamma_list, train_scores[0,:])
    plt.plot(gamma_list, test_scores[1,:])
    plt.plot(gamma_list, train_scores[1,:])
    plt.plot(gamma_list, test_scores[2,:])
    plt.plot(gamma_list, train_scores[2,:])
    plt.plot(gamma_list, test_scores[3,:])
    plt.plot(gamma_list, train_scores[3,:])
    plt.plot(gamma_list, test_scores[4,:])
    plt.plot(gamma_list, train_scores[4,:])
    plt.legend(['Deg = 1, Test', 'Deg = 1, Train', 'Deg = 2, Test', 'Deg = 2, Train', 'Deg = 3, Test', 'Deg = 3, Train', 'Deg = 4, Test', 'Deg = 4, Train', 'Deg = 5, Test', 'Deg = 5, Train'])
    plt.xlabel('Gamma')
    plt.ylabel('Mean accuracy')
    plt.show()

if __name__ == "__main__":
    features, labels, le = load_data()

    r = 2022 # random state seed
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=r, shuffle=True)

    # scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    search_KD(X_train_scaled, X_test_scaled)
    search_KNN(X_train_scaled, X_test_scaled, y_train, y_test)
    search_SVC(X_train_scaled, X_test_scaled, y_train, y_test)