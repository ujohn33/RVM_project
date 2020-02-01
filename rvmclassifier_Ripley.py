import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from math import sqrt

def linksigmoid(x):
    return 1 / (1+np.exp(-x))

def rvmclassifier(X, Y, kernel, a=1e-2, b=1e-2, c=1e-2, d=1e-2):
    N = len(X)
    T = np.matrix(Y).T
    indexes = list(range(N))
    # Based on the notes below equation (4)
    Theta = np.matrix([[kernel(X[n], X[i]) for i in range(N)] + [1] for n in range(N)])
    # I do not uderstand what these should be initialised, especially regarding the
    # note in paragrpah 2.1, from the middle of page 214 to the middle of page 215
    alphas = np.ones(N+1)*1e-4 #np.random.gamma(a, b, N)

    def y(idx, w):
        return sum([w[i]*Theta.A[idx][i] for i in range(len(indexes))])

    def objective(w):
        wMat = np.matrix(w)
        ys = np.dot(Theta, w)
        betas = linksigmoid(ys)
        pos = np.where(T == 1)[0]
        neg = np.where(T == 0)[0]
        s = np.sum(np.log(betas[0,pos])) + np.sum(np.log(1-betas[0,neg]))
        r =  - 0.5 * wMat * A * wMat.T
        return -float(s + r)
    def hessian(w):
        yn = np.array(np.dot(Theta, w).T)
        betas = np.matrix(yn*(1-yn)).A1
        return Theta.T * np.diag(betas) * Theta + A
    def jac(w):
        wMat = np.matrix(w)
        betas = linksigmoid(np.dot(Theta, w)).T
        jac = A.dot(wMat.T) - Theta.T.dot(T - betas)
        return jac.A1

    w = np.zeros(N+1)
    w0 = True

    for it in range(1000):
        #print(it)
        # Based on equations (12) and (13)
        bef = alphas.copy()
        A = np.diag(alphas)
        w = optimize.minimize(objective, w, hess=hessian, jac=jac, method="Newton-CG", options={"maxiter": 50}).x
        #print(w)
        r = [float(linksigmoid(y(i, w))*(1-linksigmoid(y(i, w)))) for i in range(N)]
        #print("hi", r)
        Beta = np.diag(r)
        A = np.diag(alphas)
        Sigma = (Theta.T * Beta * Theta + A).I

        toPrune = []
        for i in range(len(alphas)):
            # Based on equations (16) and (17)
            gamma = 1 - alphas[i]*Sigma.A[i][i]
            alphas[i] = gamma/(w[i]**2)
            if alphas[i] > 1e4:
                toPrune += [i]
                if i == len(alphas) - 1:
                    w0 = False

        diff = np.abs(bef - alphas)
        #print(alphas)
        #print(diff)
        indexes = list(np.delete(indexes, toPrune))
        alphas = np.delete(alphas, toPrune)
        Theta = np.delete(Theta, toPrune, 1)
        w = np.delete(w, toPrune, 0)

        if max(diff) < 1e-2:
            break
    #print("iteration", it)

    # Based on equations (12) and (13)
    A = np.diag(alphas)
    Sigma = (Theta.T * Beta * Theta + A).I
    w = Sigma * Theta.T * Beta * T

    #print(indexes)
    #print(w)

    def ret(x):
        addW0 = [1] if w0 else []
        theta = np.matrix([kernel(x, X[indexes[i]]) for i in range(len(indexes))] + addW0).T
        mean = w.T*theta
        sigma2 = theta.T*Sigma*theta
        return float(mean)
        #print(mean, sigma2)
        v = st.norm.pdf(0.5, mean, sigma2)
        #print(v)
        return float(v)

    return (Sigma, ret, indexes)


if __name__ == "__main__":
    # here we extract the data from the Ripley's training datase
    dataset = pd.read_csv('../Ripley_training.csv')
    dataset = dataset.sample(100)
    #print(dataset)
    X = dataset.iloc[:,0:2].values # X and Y coordinates
    t = dataset.iloc[:,2].values   # class targets
    X = np.array(X)
    t = np.array(t)

    # here we extract the data from the Ripley's test dataset
    dataset2 = pd.read_csv('../Ripley_test.csv')
    dataset2 = dataset2.sample(frac=1)
    #print(dataset2)
    X_test = dataset2.iloc[:,0:2].values # X and Y coordinates
    t_test = dataset2.iloc[:,2].values   # class targets
    X_test = np.array(X_test)
    t_test = np.array(t_test)
    #print(type(X))
    #print(t)
    #print(t_test)
    kernel = lambda x1, x2: np.exp(-0.5**(-2)*np.linalg.norm(x1-x2)**2)
    nz, c, indexes = rvmclassifier(X, t, kernel)
    xs = np.linspace(-2, 2)

    #print(nz)

    z = [[c(np.array([x, y])) for x in xs] for y in xs]
    #t_train = [c(X_test[i]) for i in range(len(X_test))]
    #t_train = [0 if i < 1 else 1 for i in t_train]
    #print(t_train)
    #print([c(X[i]) for i in range(len(X))])

    plt.contour(xs, xs, z, [0.5], colors='black', linewidth=3)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=t_test, cmap='winter')
    plt.scatter(X_test[indexes, 0], X_test[indexes, 1], s=50, c='black', marker='X')

    print(nz)
    #nz = list(map(lambda x: x[0], nz))

    #rms = sqrt(mean_squared_error(t_test, t_train))
    print("Number of support vectors is ", len(nz))
    #error = 1 - accuracy_score(t_test, t_train)
    res = [c(X_test[i]) for i in range(len(X_test))]
    res2 = [(res[i]-0.5) * (1 if t_test[i]==1 else -1) > 0 for i in range(len(X_test))]
    #print(res2)
    correct = len(list(filter(lambda v: v, res2)))
    incorrect = len(list(filter(lambda v: not v, res2)))
    #print(len(nz), list(zip(res, t_test)), correct, incorrect, incorrect/(correct+incorrect))
    print(len(nz), incorrect/(correct+incorrect))
    #print("RMSE is ",rms)
    #print("Test error is ",error)


    plt.show()
