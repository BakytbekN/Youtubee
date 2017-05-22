# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import io
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re 
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
from sklearn.cross_validation import train_test_split
from pymorphy2.tokenizers import simple_word_tokenize
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes, svm, preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

feature_names=[]
stop_words =[]
with io.open("FeatureNames.txt",'r',encoding='utf8') as f:
		for line in f:
			feature_names.append(line)
with io.open("stopwords_ru.txt",'r',encoding='utf8') as f:
		for line in f:
			stop_words.append(line)
#stop_words.extend(['???', '???', '???', '???', '????', '???', '?', '?', '??'])
train_data_df  = pd.read_csv("TestFile.txt", header=None, delimiter="\t", quoting=3)
train_data_df.columns = ["Sentiment","Text"]
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
		
def stem_tokens(token, stemmer):
    stemmed = []
    wordForm=stemmer.parse(token)[0]
#    print("VectorizeCalled")
    if token=="не" or ((len(token)!=1 and token[0].isalnum()) and (not ({'NPRO'} in wordForm.tag or{'CONJ'} in wordForm.tag or {'PREP'} in wordForm.tag or {'PRED'} in wordForm.tag or {'PRCL'} in wordForm.tag or {'INTJ'} in wordForm.tag ))):
        stemmed.append(stemmer.parse(token)[0].normal_form)
#        with io.open("FeatureNames2.txt",'a',encoding='utf8') as f:
#            f.write(stemmer.parse(token)[0].normal_form+"\t"+token+"\r\n")
    return stemmed

def tokenize(text):
    stems=[]
    morph = pymorphy2.MorphAnalyzer()
    # split by space
    text=simple_word_tokenize(text)
    # tokenize
#    print("TokenizeCalled")
    for tokens in text:
        if not is_number(tokens) and not (tokens in stop_words):
            stems.extend(stem_tokens(tokens, morph))
    return stems

	
X_train, X_test, y_train, y_test  = train_test_split(
        feature_names[0:len(train_data_df)], 
        train_data_df.Sentiment,
        train_size=0, 
        random_state=1234)
train_data_df.Text[0]=tokenize(train_data_df.Text[0])
train_data_df.Text[1]=tokenize(train_data_df.Text[1])
clf = joblib.load('SVM.pkl') 
y_pred=clf.predict(X_test)
print(classification_report(y_test, y_pred))