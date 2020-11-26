import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

n_rows=10
n_cols=10
eg=np.random.binomial(1,p=0.05,size=(n_rows,n_cols))
""" Normal Arrays are of quite high sizes """
print(eg.nbytes)

""" Sparse Matrix are of Quite Small Size """
""" One_Code_Encoders """
sp_eg=sparse.csr_matrix(eg)
print(sp_eg.data.size)

""" Usage of OneHotEncoder and Sparse Matrix """
ohe=OneHotEncoder(sparse=False)
ohe_eg_no_sparse=ohe.fit_transform(eg)

print(ohe_eg_no_sparse.data.nbytes)

ohe_n=OneHotEncoder(sparse=True)
ohe_eg_ye_sparse=ohe_n.fit_transform(eg)

print(ohe_eg_ye_sparse.data.nbytes)