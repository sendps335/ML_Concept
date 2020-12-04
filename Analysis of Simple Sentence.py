import nltk
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet

sentence=str(input('Enter the Sentence to Analyse'))
token_sentence=word_tokenize(sentence)
print(token_sentence)

important_sentence=token_sentence[:]
print(important_sentence)

sr=stopwords.words('english')
print(len(sr))
#for i in important_sentence:
#    if i in sr:
#        important_sentence.remove(i)


for i in range(0,len(important_sentence)):
    print(important_sentence[i],end=" ")
    syn=wordnet.synsets(important_sentence[i])
    if len(syn)==0:
        print('No Definition for this particular words')
    else:
        print('Definition',end="=")
        print(syn[0].definition())


sent_freq=nltk.FreqDist(important_sentence)
for key,value in sent_freq.items():
    print(key,sent_freq[key])
