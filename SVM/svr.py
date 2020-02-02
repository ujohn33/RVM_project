import numpy as np
import math
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def svr(X, Y, kernel, C=1.0, epsilon=.1):
    N = len(X)
    xMean = np.mean(X)
    xStd = np.std(X)
    yMean = np.mean(Y)
    yStd = np.std(Y)

    X = (X - xMean) / xStd
    Y = (Y - yMean) / yStd

    gramMatrix = [[kernel(X[i], X[j]) for j in range(N)] for i in range(N)]
    def objective(beta):
        betaTotal = -0.5 * np.sum(np.sum(np.dot(beta, (np.dot(beta, gramMatrix)))))
        betaTotal += np.sum([Y[i] * beta[i] for i in range(N)])
        betaTotal -= np.sum([np.abs(beta[i])*epsilon for i in range(N)])
        return -betaTotal
    bounds = [(-C, C) for i in range(N)]
    constraint = {"type": "eq", "fun": lambda beta: np.sum(beta)}
    betas = np.zeros(N)
    betas = minimize(objective, np.zeros(N), bounds=bounds, constraints=constraint).x
    nonzero = []
    print(betas)
    for i in range(N):
        if np.abs(betas[i]) > 1e-5:
            nonzero += [(betas[i], i)]
    first = nonzero[0]
    print(first)
    b = -sum([betas[i] * kernel(X[first[1]], X[i]) for i in range(N)]) + Y[first[1]]
    print(b)

    def classifier(x):
        s = 0
        for (beta, i) in nonzero:
            s += beta * kernel(X[i], (x - xMean) / xStd)

        return (s + b) * yStd + yMean
    return (nonzero, classifier)



if __name__ == "__main__":
    # generate data
    noise = np.random.normal(0,1,50)
    X = np.random.uniform(0, 10, 50)
    # function we build a regression for
    Y = X**2 - 2*X + 3 + noise

    kernel = lambda x1, x2: (x1*x2 + 1)**2
    #kernel = lambda x1, x2: np.exp(-0.5*np.abs(x1-x2)**2)


    nz, classifier = svr(X, Y, kernel, epsilon=0.1)
    xs = np.linspace(0, 10)

    z = [classifier(x) for x in xs]

    #clf = SVR(lambda x,y: np.matrix([[kernel(x[j][0], y[i][0]) for i in range(50)] for j in range(50)]))
    #clf.fit(X.reshape(-1, 1), Y)
    #z = clf.predict(xs.reshape(-1, 1))

    plt.plot(xs, z)
    plt.scatter(X,Y)
    plt.show()

