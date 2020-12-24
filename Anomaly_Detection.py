import numpy as np
import pandas as pd
from sklearn import svm
from matplotlib import pyplot as plt

X_train=np.sort(25*np.random.rand(200,1),axis=0)
Y_train=np.cos(X_train).ravel()
Y_train[::4]+=4*(0.5-np.random.rand(50))

sv=svm.SVR(kernel='rbf',C=1e3,gamma=0.1)
sv.fit(X_train,Y_train)
Y_pred=sv.predict(X_train)

error=Y_pred-Y_train
upper=np.mean(error)+2.41*np.std(error)
lower=np.mean(error)-2.41*np.std(error)
anomaly=np.concatenate([np.where(error>upper)[0],np.where(error<lower)[0]])

plt.figure(figsize=(9,5))
plt.scatter(X_train,Y_train,color='pink',label='Dataset')
plt.scatter(X_train[anomaly],Y_train[anomaly],marker='*',color='green',label='anomaly',s=120)
#plt.hold('on')
plt.plot(X_train,Y_pred,color='yellow',label='curve')
plt.xlabel('X_Labels')
plt.ylabel('Y_Labels')
plt.title('Gradient Boost Classifier')
plt.legend(loc=3,frameon=False)
plt.show()