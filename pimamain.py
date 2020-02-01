import numpy as np
from svm.svm import svm
from rvm.rvmclassifier import rvmclassifier
import pima.datasets as pima


Xtrain, Ytrain, xMean, xScale = pima.getTrainingData()
Xtest, Ytest = pima.getTestData(xMean, xScale)

gaussianKernel = lambda sigma2: lambda x1, x2: np.exp((-np.linalg.norm(x1-x2)**2)/(2*sigma2))

def runsvm():
    print("SVM:")
    kernel = gaussianKernel(5)

    pos = np.where(Ytrain == True)[0]
    neg = np.where(Ytrain == False)[0]

    Y = np.zeros(Ytrain.shape)
    Y[pos] = 1
    Y[neg] = -1

    nz, classifier = svm(Xtrain, Y, kernel, C=0.9)

    res = [classifier(Xtest[i]) for i in range(len(Xtest))]
    res2 = [res[i] * (1 if Ytest[i] else -1) > 0 for i in range(len(Xtest))]
    #print(res2)
    correct = len(list(filter(lambda v: v, res2)))
    incorrect = len(list(filter(lambda v: not v, res2)))
    print(len(nz), list(zip(res, Ytest)), correct, incorrect, incorrect/(correct+incorrect))

    # 0.01 => 200 vectors 221 : 111
    # 0.05 => 180 vectors 223 : 109
    # 0.1  => 167 vectors 224 : 108
    # 0.12 => 167 vectors 221 : 111
    # 0.25 => 163 vectors 228 : 104
    # 0.5  => 166 vectors 230 : 102
    # 0.75 => 171 vectors 226 : 106
    # 1.0  => 175 vectors 228 : 104
    # 2.0  => 179 vectors 233 : 99
    # 3.0  => 175 vectors 233 : 99
    # 5.0  => 167 vectors 240 : 92
    # 7.0  => 166 vectors 234 : 98
    # 10.0 => 167 vectors 233 : 99

    # 5.0 =>                            moreits:
    # C = 0.5 => 164 vectors 228 : 104    
    # C = 0.7 => 164 vectors 231 : 101    
    # C = 0.9 => 167 vectors 240 : 92     150 vectors 239 : 93
    # C = 1.0 => 167 vectors 240 : 92     
    # C = 2.0 => 170 vectors 235 : 97     
    # C = 5.0 => 170 vectors 235 : 97     

def runrvm():
    print("RVM")
    kernel = gaussianKernel(1)

    pos = np.where(Ytrain == True)[0]

    Y = np.zeros(Ytrain.shape)
    Y[pos] = 1
    Y = list(Y)

    nz, classifier = rvmclassifier(Xtrain, Y, kernel)

    print(nz)
    res = [classifier(Xtest[i]) for i in range(len(Xtest))]
    res2 = [(res[i]-0.5) * (1 if Ytest[i] else -1) > 0 for i in range(len(Xtest))]
    #print(res2)
    correct = len(list(filter(lambda v: v, res2)))
    incorrect = len(list(filter(lambda v: not v, res2)))
    print(len(nz), list(zip(res, Ytest)), correct, incorrect, incorrect/(correct+incorrect))

    # 0.1  => 22 vectors 212 : 120
    # 1.0  =>  8 vectors 226 : 106
    # 5.0  =>  3 vectors 223 : 109

runrvm()