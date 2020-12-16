import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

class Feature_Selection:
    def __init__(self,n_features,problem_type,scoring):
        valid_scoring=dict()
        if problem_stype == 'classification':
            valid_scoring['f_classif']=f_classif
            valid_scoring['chi2']=chi2
            valid_scoring['mutual_info_classif']=mutual_info_classif
        else:
            valid_scoring['f_regression']=f_regression
            valid_scoring['mutual_info_regression']=mutual_info_regression
        
        if scoring not in valid_scoring:
            raise Exception('Invalid Scoring Type')
        
        if isinstance(n_features,int):
            self.selection=SelectKBest(valid_scoring[scoring],k=n_features)
        elif isinstance(n_features,float):
            self.selection=SelectPercentile(valid_scoring[scoring],percentile=int(100*n_features))
        else:
            raise Exception('Invalid Type of Feature')
    
    def fit(self,x,y):
        return self.selection.fit(X,y)
    
    def transform(self,X):
        return self.selection.transform(X)
    
    def fit_transform(self,X,y):
        return self.selection.fit_transform(X,y)
        

X=[[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
X0=[[np.nan, 6, np.nan], [2, 5, np.nan], [1,5,3], [7, np.nan, 4]]

ki=KNNImputer(n_neighbors=2)
X=ki.fit_transform(X)
X0=ki.transform(X0)

iris=datasets.load_iris()
df_iris=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df_iris['Class']=iris.target

print(df_iris.drop('Class',axis=1,inplace=False).corr())
