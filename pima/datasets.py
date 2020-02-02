import os
import numpy as np

loc = os.path.join(os.getcwd(), os.path.dirname(__file__))

def getData(file):
    X = []
    Y = []

    file.readline() # Discard the title row
    data = file.read()
    rows = data.splitlines(False)
    for r in rows:
        row = list(filter(lambda s: s != "", r.split(" ")))
        X += [list(map(float, row[:-1]))]
        Y += [row[-1] == "Yes"]
    X = np.array(X)
    Y = np.array(Y)
    
    return (X, Y)

def getTrainingData():
    with open(os.path.join(loc, "train")) as file:
        X, Y = getData(file)

        xMean = []
        xScale = []
        for i in range(X.shape[1]):
            mu = np.mean(X[:, i])
            sc = np.var(X[:, i])
            X[:, i] -= mu
            X[:, i] /= sc

            xMean += [mu]
            xScale += [sc]
        
        return (X, Y, xMean, xScale)

def getTestData(xMean, xScale):
    with open(os.path.join(loc, "test")) as file:
        X, Y = getData(file)

        for i in range(X.shape[1]):
            mu = np.mean(X[:, i])
            sc = np.var(X[:, i])
            X[:, i] -= xMean[i]
            X[:, i] /= xScale[i]
        
        return (X, Y)

