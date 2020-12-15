import numpy as np
import pandas as pd
import operator
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score


""" Loading of Datasets """
iris=datasets.load_iris()
df_iris=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df_iris['Class']=iris.target
#print(df_iris.head())
""" ScikitLearn is Only Used for Creation of the Dataset and the Splitting of the dataset"""


""" KNN from Basics """
def squared_error_distance(testing_data,training_data,test_features):
    sq_d=0
    for i in range(0,test_features):
        ##Squared_Error_Distances
        ##Of Each Features
        sq_d+=((testing_data[i]-training_data[i])**2)
    sq_d=sq_d**0.5
    return sq_d

def KNN(training_set,testing_set,K):
    #taking the dimensions into account
    test_size=testing_set.shape[0]
    test_features=testing_set.shape[1]
    test_work=[]
    train_size=training_set.shape[0]
    """ Each Testing Set """
    ###Storing Distances of Each Testing Sets from the Training Sets
    for j in range(0,test_size):
        ##Dictionary of Distance of Each testing set
        ##From all the Training Sets
        distance_dict=dict()
        for i in range(0,train_size):
            dd=squared_error_distance(testing_set.iloc[j],training_set.iloc[i],test_features)
            distance_dict[i]=dd
        ###Sorting the Items 
        sorted_distance_list=sorted(distance_dict.items(),key=operator.itemgetter(1))
        
        neighbours=[]
        for i in range(0,K):
            ##Appending the Nearest K Points from the Particular Test Set
            ##i.e. from the jth Testing Set
            neighbours.append(sorted_distance_list[i][0])

        votes=dict()
        for i in range(0,len(neighbours)):
            ## Response is the Label
            ##That the Particular Point is classified
            ##in the Training Set
            response=training_set.iloc[neighbours[i]][-1]
            if response not in votes.keys():
                votes[response]=1
            else:
                votes[response]+=1
        ##Sort in Descending Order
        sorted_votes_list=sorted(votes.items(),key=operator.itemgetter(1),reverse=True)
        ##Store the Particular Label/Class
        ##To which our jTH test set is nearer to
        test_work.append([int(sorted_votes_list[0][0]),neighbours])
        
    ##Return the Predicted Labels and the Neighbours
    ##For all the Testsets
    return test_work


""" Testing Phase """
""" Let's try on spliiting the main data set to 140-10 """

#Randomly Shuffling the Datasets
df_iris=df_iris.sample(frac=1).reset_index(drop=True)

#Spliting of the Datasets into Training Sets and Testing Sets
training_set=df_iris.iloc[:140]
testing_set=df_iris.iloc[140:]

#print(testing_set.Class)

Kmax=7
accuracys=dict()
f1_scores=dict()
for i in range(1,Kmax+1,2):
    test_work=KNN(training_set,testing_set,i)
    y_pred=[test_work[i][0] for i in range(0,len(test_work))]
    y_true=testing_set.Class.values
    accuracy=accuracy_score(y_true,y_pred)
    f1_ss=f1_score(y_true,y_pred,average=None)
    accuracys[i]=accuracy*100
    f1_scores[i]=f1_ss

plt.plot(accuracys.keys(),accuracys.values(),color='green')
plt.xlabel('No of Centroids')
plt.ylabel('Accuracy Score')
plt.show()

plt.plot(f1_scores.keys(),f1_scores.values(),color='pink')
plt.xlabel('No of Centroids')
plt.ylabel('F1 Score')
plt.show()
