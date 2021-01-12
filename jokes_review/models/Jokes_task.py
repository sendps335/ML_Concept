import nltk
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import string

df_joke=pd.read_csv(r'C:\Users\DEBIPRASAD\Desktop\Projetc Work\Jokes_Ratings\jokes.csv')
df_joke['joke_text']=df_joke['joke_text'].str.lower()

l1=list(df_joke['joke_text'])
l2=[]
for i in l1:
    #Removing the 's' from the end part of the words
    k=re.sub(r"'s\b",'',i)
    #Removing the expressions which aren't words i.e. only words are present
    k=re.sub("[^a-zA-Z]"," ",k)
    #Remove the Punctuations
    k=''.join([j for j in k if j not in string.punctuation])
    l2.append(k)

stopword=nltk.corpus.stopwords.words('english')
l3=[]
for i in l2:
    k="".join([j for j in i if i not in stopword])
    l3.append(k)
df_joke['joke_text']=l3

def tokenize(text):
    token=re.split('\W+', text)
    return token
df_joke['joke_text']=df_joke['joke_text'].apply(lambda x:tokenize(x))
lm=nltk.WordNetLemmatizer()
def lemmatizerr(text):
    l=[]
    text=list(text.split(' '))
    for i in text:
        k=lm.lemmatize(i)
        l.append(k)
    return ' '.join(l)
df_joke['joke_text']=df_joke['joke_text'].apply(lambda x:lemmatizerr(x))

def sentimentt(text):
    return (TextBlob(text).sentiment.polarity)
df_joke['sentiment_score']=df_joke['joke_text'].apply(lambda x:sentimentt(x))

print(df_joke.head())