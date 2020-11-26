import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import model_selection

def create_folds(df_data):
    df_data['kfold']=-1
    df_data.sample(frac=1).reset_index(drop=True)
    
    """ The Number of Bins as per formulae"""
    num_bins=int(np.floor(1+np.log(len(df_data))))
    
    """ Creating the Bins """
    df_data.loc[:,'bins']=pd.cut(df_data['target'],bins=num_bins,labels=False)
    
    k_f_model=model_selection.StratifiedKFold(n_splits=5)
    for f,(t_,v_) in enumerate(k_f_model.split(X=df_data,y=df_data.bins.values)):
        df_data.loc[v_,'kfold']=f
    
    """ Droping the Bins as the folds have been created """
    df_data.drop('bins',axis=1,inplace=True)
    
    """ Returning Dataframe with Folds """
    return df_data

if __name__=='__main__':
    X,y=datasets.make_regression(n_samples=1500,n_features=100,n_targets=1)
    df_data=pd.DataFrame(X,columns=['f '+str(i) for i in range(1,X.shape[1]+1)])
    df_data.loc[:,'target']=y
    
    print(X.shape)
    print()
    print(y.shape)
    print()
    print(df_data.head(4))
    
    """ Calling the Function which creates the Folds """
    print()
    df_data=create_folds(df_data)
    print(df_data.kfold.value_counts())