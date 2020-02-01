import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from math import sqrt

def svm(X, Y, kernel, C=1.0):
    N = len(X)
    gramMatrix = [[kernel(X[i], X[j]) for j in range(N)] for i in range(N)]
    #print(gramMatrix)
    P = [[gramMatrix[i][j] * Y[i] * Y[j] for j in range(N)] for i in range(N)]
    #print(P)
    def objective(alpha):
        s = sum(alpha)
        for i in range(N):
            for j in range(N):
                s -= 0.5*alpha[i]*alpha[j]*P[i][j]
        return -s
    bounds = [(0, C) for i in range(N)]
    constraint = {"type": "eq", "fun": lambda alpha: Y.dot(alpha)}
    alphas = np.zeros(N)

    alphas = optimize.minimize(objective, alphas, bounds=bounds, constraints=constraint).x

    nonzero = []
    for i in range(N):
        if alphas[i] > 1e-3:
            nonzero += [(alphas[i], i)]

    first = nonzero[0]
    b = - sum([nonzero[i][0]*Y[nonzero[i][1]]*gramMatrix[first[1]][nonzero[i][1]] for i in range(len(nonzero))]) + Y[first[1]]

    def classifier(x):
        s = 0
        for (alpha, i) in nonzero:
            s += alpha * Y[i] * kernel(X[i], x)
        return s + b

    return (nonzero, classifier)

def fivefold(X, t, kernel):
    # 5-fold cross validation
    kf = KFold(n_splits=5)
    C_list = [1, 2, 3, 4, 5]
    i = 0
    res = np.zeros(5)
    for train_index, test_index in kf.split(X):
        Xf_train, Xf_test = X[train_index], X[test_index]
        tf_train, tf_test = t[train_index], t[test_index]
        Xf_train = np.array(Xf_train)
        Xf_test = np.array(Xf_test)
        tf_train = np.array(tf_train)
        tf_test = np.array(tf_test)
        #print(Xf_train)
        #print(len(Xf_train))
        nzf, cf = svm(Xf_train, tf_train, kernel, C_list[i])
        t_prediction = [cf(Xf_test[j]) for j in range(len(Xf_test))]
        t_prediction = [-1 if i < 0 else 1 for j in t_prediction]
        res[i] =  1 - accuracy_score(tf_test, t_prediction)
        i+=1
    argmin = np.argmin(res)
    print(res)
    return C_list[argmin]



if __name__ == "__main__":
    # here we extract the data from the Ripley's training dataset
    dataset = pd.read_csv('../Ripley_training_prep.csv')
    dataset = dataset.sample(100)
    print(dataset)
    X = dataset.iloc[:,0:2].values # X and Y coordinates
    t = dataset.iloc[:,2].values   # class targets
    print(X)
    print(t)



    X = np.array(X)
    t = np.array(t)
    # here we extract the data from the Ripley's training dataset
    dataset2 = pd.read_csv('../Ripley_test_prep.csv')
    dataset2 = dataset2.sample(frac=1)
    X_test = dataset2.iloc[:,0:2].values # X and Y coordinates
    t_test = dataset2.iloc[:,2].values   # class targets
    X_test = np.array(X_test)
    t_test = np.array(t_test)
    #print(type(X))
    #print(t)
    #print(t_test)
    kernel = lambda x1, x2: np.exp(-0.5**(-2)*np.linalg.norm(x1-x2)**2)
    # find best C from 5-fold cross-validation
    #C = fivefold(X,t,kernel)
    C = 2
    print("Best C is ", C)
    nz, c = svm(X, t, kernel, C)
    #xs = np.linspace(-4, 4)

    #print(nz)
    xs = np.linspace(-2, 2)
    ys = np.linspace(-2, 2)
    #print(X_test)
    z = [[c(np.array([x, y])) for x in xs] for y in xs]
    t_train = [c(X_test[i]) for i in range(len(X_test))]
    t_train = [-1 if i < 0 else 1 for i in t_train]
    #z = [c(X_test[i]) for i in range(len(X_test))]
    #print(t_train)
    #print(t_test)
    plt.contour(xs, ys, z, [0.5], colors='black', linewidth=3)
    #plt.scatter(list(map(lambda x: x[0], X)), list(map(lambda x: x[1], X)), c=t, cmap='winter')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=t_test, cmap='winter')
    nz = list(map(lambda x: x[1], nz))
    plt.scatter(X_test[nz, 0], X_test[nz, 1], s=50, c='black', marker='X')
    #rms = sqrt(mean_squared_error(t_test, t_train))
    print("Number of support vectors is ", len(nz))
    error = 1 - accuracy_score(t_test, t_train)
    #print("RMSE is ",rms)
    print("Test error is ",error)


    plt.show()
