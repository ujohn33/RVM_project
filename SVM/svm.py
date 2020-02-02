import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def svm(X, Y, kernel, C=1.0):
    N = len(X)
    gramMatrix = [[kernel(X[i], X[j]) for j in range(N)] for i in range(N)]
    print(gramMatrix)
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

    alphas = optimize.minimize(objective, alphas, bounds=bounds, constraints=constraint, options={"maxiter":20}).x

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

    print(alphas)

    return (nonzero, classifier)


if __name__ == "__main__":
    X = np.array([[1, 1],[1,2],[-1,3],[-1,4], [-1.1, 3]])
    # kernel = lambda x1, x2: np.exp(-0.5**(-2)*np.abs(x1-x2)**2)
    kernel = lambda x1, x2: np.exp(-0.5**(-2)*np.abs(x1-x2)**2)
    nz, c = svm(X, np.array([1,1,-1,-1, 1]), kernel, C=1)
    #print(type(X))
    #print(type(np.array([1,1,-1,-1, 1])))
    xs = np.linspace(-4, 4)

    #print(nz)

    z = [[c(np.array([x, y])) for x in xs] for y in xs]

    plt.contour(xs, xs, z, (-1, 0, 1))
    plt.scatter(list(map(lambda x: x[0], X)), list(map(lambda x: x[1], X)))
    nz = list(map(lambda x: x[1], nz))

    #print(X[nz])

    #plt.show()
