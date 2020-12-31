import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import model_selection
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import preprocessing

df_train=pd.read_csv(r'C:\Users\DEBIPRASAD\Desktop\Projetc Work\nlp_disaster_tweets\train.csv')
df_test=pd.read_csv(r'C:\Users\DEBIPRASAD\Desktop\Projetc Work\nlp_disaster_tweets\test.csv')

df_train.keyword.fillna('None',inplace=True)
df_test.keyword.fillna('None',inplace=True)

df_train.location.fillna('None',inplace=True)
df_test.location.fillna('None',inplace=True)

""" Randomly Shuffle the Training Datasets"""
df_train.sample(frac=1).reset_index(drop=True)

"""End"""

""" Hashtags """
train_text=[]
for i in range(df_train.shape[0]):
    kk=df_train['text'][i]+' '+df_train['keyword'][i]
    kk=kk.replace('#',' ')
    train_text.append(kk)
df_train['text']=train_text


test_text=[]
for i in range(df_test.shape[0]):
    kk=df_test['text'][i]+' '+df_test['keyword'][i]
    kk=kk.replace('#',' ')
    test_text.append(kk)
df_test['text']=test_text

""" End of Hashtag """

""" Label Encoding """
"""
lb1=preprocessing.LabelEncoder()
lb1.fit(df_train.location.values)
lb1_train=lb1.transform(df_train.location.values)
lb1_test=lb1.transform(df_test.location.values)
df_train['location']=lb1_train
df_test['location']=lb1_test
"""

lb2=preprocessing.LabelEncoder()
lb2.fit(df_train.keyword.values)
lb2_train=lb2.transform(df_train.keyword.values)
lb2_test=lb2.transform(df_test.keyword.values)
df_train['keyword']=lb2_train
df_test['keyword']=lb2_test

""" End Of Encoding """
test_id=df_test.id

df_train['kfold']=-1
kf=model_selection.StratifiedKFold(n_splits=5)
for f,(t_,v_) in enumerate(kf.split(X=df_train.text,y=df_train.target)):
    df_train.loc[v_,'kfold']=f

test_sets=[]
for fold in range(5):
    df_train_t=df_train[df_train['kfold'] != fold]
    df_cross=df_train[df_train['kfold'] == fold]
    
    xtrain=df_train_t.text.values
    xcross=df_cross.text.values
    xtest=df_test.text.values
    
    ytrain=df_train_t.target.values
    ycross=df_cross.target.values
    
    tfd=CountVectorizer(tokenizer=word_tokenize,token_pattern=None)
    tfd.fit(xtrain)
    
    xtrain=tfd.transform(xtrain)
    xcross=tfd.transform(xcross)
    xtest=tfd.transform(xtest)
    
    nb=naive_bayes.MultinomialNB()
    nb.fit(xtrain,ytrain)
    ytrain_pred=nb.predict(xtrain)
    ycross_pred=nb.predict(xcross)
    ytest_pred=nb.predict(xtest)
    
    accuracy1=metrics.accuracy_score(ytrain,ytrain_pred)
    accuracy2=metrics.accuracy_score(ycross,ycross_pred)
    test_sets.append([fold,ytest_pred])
    
    print(f"Fold = {fold}")
    print(f"Training Set Accuracy = {accuracy1}")
    print(f"Cross-Validation Set Accuracy = {accuracy2}")
    print()

""" Almost End """