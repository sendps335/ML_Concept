""" Task 1 """
""" SC Task """
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot as plt
import seaborn as sns


def cross_validation(df_train,df_test,fold):
    """ Model_Selection """
    """ Hyperparameter Tuning """
    classifier=ensemble.RandomForestClassifier(n_jobs=-1)
    params={'max_depth':np.arange(1,30),'n_estimators':np.arange(100,1500,100),'criterion':['gini','entropy']}
    model=RandomizedSearchCV(estimator=classifier,param_distributions=params,verbose=10,n_iter=20,scoring='accuracy',n_jobs=5,cv=5)
    
    """ Shuffling the Data """
    df_train.sample(frac=1).reset_index()
    
    """ Sorting it Out """
    X=df_train.drop('Class',axis=1,inplace=False).values
    y=df_train.Class.values
    x_test=df_test.values
    
    """ Sturge's Rule """
    num_bins=int(np.floor((1+np.log(df_train.shape[0]))/2))
    
    """ Stratified K_Model """
    kf=StratifiedKFold(n_splits=fold)
    
    train_acc=[]
    cross_acc=[]
    test_model=[]
    
    for train_index,cross_v_index in kf.split(X=X,y=y):
        """ Training Set"""
        x_train=X[train_index]
        y_train=y[train_index]
        
        """ Cross-Validation Set """
        x_cross=X[cross_v_index]
        y_cross=y[cross_v_index]
        
        """ Model Selection """
        model.fit(x_train,y_train)
        y_train_pred=model.predict(x_train)
        y_cross_pred=model.predict(x_cross)
        
        train_acc.append(metrics.accuracy_score(y_train,y_train_pred))
        cross_acc.append(metrics.accuracy_score(y_cross,y_cross_pred))
        
        y_test=model.predict(x_test)
        test_model.append(y_test)
        
    return train_acc,cross_acc,test_model

def visualization(train_accuracy,cross_validation_accuracy,fold):
    m=len(train_accuracy)
    mm=np.arange(1,m+1)
    plt.scatter(train_accuracy,mm,color='cyan',marker='*',label='Train Accuracy')
    plt.scatter(cross_validation_accuracy,mm,color='pink',marker='^',label='Cross Validation Accuracy')
    plt.plot(train_accuracy,mm,color='cyan',label='Train Curve')
    plt.plot(cross_validation_accuracy,mm,color='pink',label='Cross Validation Curve')
    plt.xlabel('Size')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='lower right')
    plt.title('Visualization Result for '+str(fold)+'th Fold')
    plt.show()
    
if __name__=='__main__':
    X,y=datasets.make_classification(n_samples=3000,n_features=35,n_classes=4,n_clusters_per_class=2,n_informative=8)
    
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
    df_train=pd.DataFrame(x_train,columns=["Feature_"+str(i) for i in range(1,len(x_train[0])+1)])
    df_train['Class']=y_train
    
    df_test=pd.DataFrame(x_test,columns=["Feature_"+str(i) for i in range(1,len(x_test[0])+1)])
    #df_test['Class']=y_test
    
    max_fold=4
    for i in range(2,max_fold+1):
        train_accuracy,cross_validation_accuracy,test_models=cross_validation(df_train,df_test,i)
        print("Results for Fold"+str(i))
        visualization(train_accuracy,cross_validation_accuracy,i)
    
    """ End Of The Programme """