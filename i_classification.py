import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import os

first=plt.imread(r'C:\Users\DEBIPRASAD\Desktop\dps\th.jpg')
print(np.shape(first))
print(np.min(first),np.max(first))

shap=np.shape(first)
pixel=np.reshape(first,(shap[0]*shap[1],shap[2]))
print(pixel.shape)

plt.hist2d(pixel[:,1],pixel[:,2],bins=(5,5))
plt.show()

plt.imshow(first)
plt.show()

""" Second One """
second=plt.imread(r'C:\Users\DEBIPRASAD\Desktop\dps\th (1).jpg')
print(np.shape(second))
print(np.min(second),np.max(second))

shap2=np.shape(second)
pixel2=np.reshape(second,(shap2[0]*shap2[1],shap2[2]))
print(pixel2.shape)

plt.hist2d(pixel2[:,1],pixel2[:,2],bins=(5,5))
plt.show()

plt.imshow(second)
plt.show()

###

from sklearn import cluster
cm=cluster.KMeans(7)
cm.fit(pixel)
clustered_1=cm.predict(pixel)
clustered_1=np.reshape(clustered_1,(shap[0],shap[1]))
plt.imshow(clustered_1)
plt.show()

cm2=cluster.KMeans(10)
cm2.fit(pixel2)
clustered_2=cm2.predict(pixel2)
clustered_2=np.reshape(clustered_2,(shap2[0],shap2[1]))
plt.imshow(clustered_2)
plt.show()
