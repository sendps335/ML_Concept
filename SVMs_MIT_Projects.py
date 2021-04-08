import numpy as np
import math
from scipy.special import softmax

def svm(theta,X,C,beta):
    ypred=np.dot(X,theta.T)+beta
    res=(0.5)*np.sum(np.dot(theta.T,theta))+C*beta
    return res,ypred

def softmax1(ypred):
    e=np.exp(ypred)
    prob=e/np.sum(e)
    return prob

def softmax2(ypred):
    return softmax(ypred)

X=np.array([[1,2,3],[4,5,6]])
theta=np.array([[0.77,0.332,2.334]])
beta=5
res,y=svm(theta,X,0.05,beta)
print(res,y)

print(softmax1([0,1,0]))
print(softmax2([0,1,0]))
print(sum(softmax1([0,1,0])))
print(sum(softmax2([0,1,0])))
