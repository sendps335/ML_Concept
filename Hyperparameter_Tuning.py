import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm

bc=datasets.load_breast_cancer()
df_bc=pd.DataFrame(data=bc.data,columns=bc.feature_names)
df_bc['Result']=bc.target

print(df_bc.Result.value_counts())

""" Using SVM Model Without Tuning the HyperParameter """
x_train,x_test,y_train,y_test=train_test_split(df_bc.drop('Result',axis=1,inplace=False).values,df_bc.Result.values,test_size=0.3,random_state=101)
sv=svm.SVC()
sv.fit(x_train,y_train)
train_predictions=sv.predict(x_train)
test_predictions=sv.predict(x_test)

print(classification_report(y_train,train_predictions))
print(classification_report(y_test,test_predictions))

"""  End """

""" Model after Tuning the Hyperparameters """

parameters={'C':[0.01,0.1,1,10,100,1000],'kernel':['rbf'],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid_sv=GridSearchCV(estimator=svm.SVC(),param_grid=parameters,verbose=3)

grid_sv.fit(x_train,y_train)
train_predictions=grid_sv.predict(x_train)
test_predictions=grid_sv.predict(x_test)

print(classification_report(y_train,train_predictions))
print(classification_report(y_test,test_predictions))

print(grid_sv.best_params_)
print(grid_sv.best_estimator_)

""" Continued """
from sklearn import ensemble
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics

X,y=datasets.make_classification(n_samples=1000,n_features=20,n_classes=4,n_clusters_per_class=1,n_informative=6)
df=pd.DataFrame(X,columns=['feature_'+ str(i) for i in range(1,len(X[0])+1)])
df['Target']=y

print(df.shape)
vt=VarianceThreshold(threshold=0.7)
vt.fit(X)
X=vt.transform(X)
print(df.shape)

classifier=ensemble.RandomForestClassifier(n_jobs=-1)
params={'n_estimators':np.arange(1,100),'max_depth':np.arange(1,15),'criterion':['gini','entropy']}
model=RandomizedSearchCV(estimator=classifier,param_distributions=params,n_jobs=5,n_iter=20,scoring='accuracy',verbose=10,cv=5)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)
model.fit(x_train,y_train)
y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)

print(model.best_params_)
print(model.best_estimator_)

print(metrics.accuracy_score(y_train,y_train_pred))
print(metrics.accuracy_score(y_test,y_test_pred))
print(metrics.f1_score(y_train,y_train_pred,average='weighted'))
print(metrics.f1_score(y_test,y_test_pred,average='weighted'))

""" Without Hyperparameter Tuning """
rf=ensemble.RandomForestClassifier()
rf.fit(x_train,y_train)
y_train_rf=rf.predict(x_train)
y_test_rf=rf.predict(x_test)

print(metrics.accuracy_score(y_train,y_train_rf))
print(metrics.accuracy_score(y_test,y_test_rf))
print(metrics.f1_score(y_train,y_train_rf,average='weighted'))
print(metrics.f1_score(y_test,y_test_rf,average='weighted'))
