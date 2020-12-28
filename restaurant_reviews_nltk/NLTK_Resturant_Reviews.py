import numpy as np
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn import naive_bayes

df=pd.read_csv(r'C:\Users\DEBIPRASAD\Desktop\Git\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
print(df.head())

""" Folds """
df['kfold']=-1
kf=model_selection.StratifiedKFold(n_splits=5)
y=df.Liked.values
for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
    df.loc[v_,'kfold']=f

""" Models """
for fold in range(5):
    df_train=df[df['kfold'] != fold].reset_index(drop=True)
    df_cross=df[df['kfold'] == fold].reset_index(drop=True)
    
    tfid_cv=TfidfVectorizer(tokenizer=word_tokenize,token_pattern=None)
    tfid_cv.fit(df_train.Review.values)
    xtrain=tfid_cv.transform(df_train.Review.values)
    xcross=tfid_cv.transform(df_cross.Review.values)
    
    ytrain=df_train.Liked.values
    ycross=df_cross.Liked.values
    
    nb=naive_bayes.MultinomialNB()
    nb.fit(xtrain,ytrain)
    ytrain_pred=nb.predict(xtrain)
    ycross_pred=nb.predict(xcross)
    
    accuracy1=metrics.accuracy_score(ytrain,ytrain_pred)
    accuracy2=metrics.accuracy_score(ycross,ycross_pred)
    f1=metrics.f1_score(ycross,ycross_pred)
    
    print(f"Fold = {fold}")
    print(f"Training Set Accuracy = {accuracy1}")
    print(f"Cross-Validation Set Accuracy = {accuracy2}")
    print(f"Cross-Validation F1 Score = {f1}")
    print("")
    
    