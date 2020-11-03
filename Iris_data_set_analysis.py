import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics

iris=datasets.load_iris()

iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_data['Class']=iris['target']
iris_data.columns=['sepal_length','sepal_width','petal_length','petal_width','Class']
print(iris_data.head())

print()

X_features=['sepal_length','sepal_width','petal_length','petal_width']

X=iris_data[X_features]
Y=iris_data.Class

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

clf=tree.DecisionTreeClassifier(max_depth=3)
clf2=RandomForestClassifier(max_depth=3)
clf.fit(X_train,Y_train)
clf2.fit(X_train,Y_train)

Y_train_pred_DT=clf.predict(X_train)
Y_test_pred_DT=clf.predict(X_test)
Y_train_pred_RF=clf2.predict(X_train)
Y_test_pred_RF=clf2.predict(X_test)

train_acc_DT=metrics.accuracy_score(Y_train,Y_train_pred_DT)*100
test_acc_DT=metrics.accuracy_score(Y_test,Y_test_pred_DT)*100
train_acc_RF=metrics.accuracy_score(Y_train,Y_train_pred_RF)*100
test_acc_RF=metrics.accuracy_score(Y_test,Y_test_pred_RF)*100

print("Decision Tree Accuracy",train_acc_DT,test_acc_DT)
print("Random Forest Accuracy",train_acc_RF,test_acc_RF)