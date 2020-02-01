import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def rvm(X, Y, kernel, a=1e-2, b=1e-2, c=1e-2, d=1e-2):
    N = len(X)
    # Based on the notes below equation (4)
    Theta = np.matrix([[kernel(X[n], X[i]) for i in range(N)] for n in range(N)])

    # I do not uderstand what these should be initialised, especially regarding the
    # note in paragrpah 2.1, from the middle of page 214 to the middle of page 215
    alphas = np.ones(N)#np.random.gamma(a, b, N)
    sigma2  = 1#1/np.random.gamma(c, d)

    indexes = list(range(N))
    for it in range(100000):
        # Based on equations (12) and (13)
        A = np.diag(alphas)
        Sigma = ((1/sigma2) * Theta.T * Theta + A).I
        mu = (1/sigma2) * Sigma * Theta.T * Y

        bef = alphas.copy()
        toPrune = []
        gammas = np.zeros(len(alphas))
        for i in range(len(alphas)):
            # Based on equations (16) and (17)
            gammas[i] = 1 - alphas[i]*Sigma.A[i][i]
            alphas[i] = gammas[i]/(mu[i]**2)
            if alphas[i] > 1e12:
                toPrune += [i]
        diff = np.abs(bef - alphas)
        indexes = list(np.delete(indexes, toPrune))
        alphas = np.delete(alphas, toPrune)
        Theta = np.delete(Theta, toPrune, 1)
        mu = np.delete(mu, toPrune, 0)
        
        # Based on equation (18)
        sigma2 = np.linalg.norm(Y - Theta * mu)**2/(N-sum(gammas))

        if max(diff) < 0.1:
            break
    print("iteration", it)
    
    # Based on equations (12) and (13)
    A = np.diag(alphas)
    Sigma = np.linalg.inv((1/sigma2) * Theta.T * Theta + A)
    mu = (1/sigma2) * Sigma * Theta.T * Y

    print(indexes)

    def ret(x):
        r = [mu.A1[i]*kernel(x, X[indexes[i]]) for i in range(len(indexes))]
        return sum(r)# + r0
    return (Sigma, ret)


if __name__ == "__main__":
    # generate data
    noise = np.random.normal(0,1,50)
    X = np.random.uniform(0, 10, 50)
    kernel = lambda x1, x2: (x1[0]*x2[0] + 1)**2
    # function we build a regression for
    Y = X**2 - 2*X + 3 + noise

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    print(X.shape,Y.shape)

    sigma, classifier = rvm(X, Y, kernel)
    print(sigma)

    xs = np.matrix(np.linspace(0, 10)).reshape(-1, 1)
    z = [classifier(x).A[0] for x in xs]
    plt.plot(xs.A, z)
    plt.scatter(X, Y)
    plt.show()