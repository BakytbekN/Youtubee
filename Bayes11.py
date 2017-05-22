import pandas as pd
import numpy as np
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import naive_bayes
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
def prettyPrint(word):
	print("#"*60)
	print(word)
	print("#"*60)
	print("="*60)

prettyPrint("reading files")
allFiles = ["DENIScutted.txt","OSIPOVcutted.txt","VADIMcutted.txt"]
trainFileName="temporal.txt"
list_ = []
for file_ in allFiles:
	with io.open(file_,'r',encoding='utf8') as f:
		prettyPrint("reading one of many Files")
		for line in f:
			list_.append(line)
with io.open(trainFileName,'w',encoding='utf8') as f:
	prettyPrint("writing to destination file")
	for w in list_:
		f.write(w)
train_data_df  = pd.read_csv(trainFileName, header=None, delimiter="\t", quoting=3)
train_data_df.columns = ["Sentiment","Text"]

test_data_file_name = 'FILEVcutted.txt'
test_data_df = pd.read_csv(test_data_file_name, header=None, delimiter="\t", quoting=3)
test_data_df.columns = ["Sentiment","Text"]
prettyPrint("panda read files")
#print(train_data_df.shape)
#print(test_data_df.shape)
#print(np.mean([len(str(s).split(" ")) for s in train_data_df.Text ]))

################################
#Reading stop wrods
stop_words =[]
with io.open("stopwords_ru.txt",'r',encoding='utf8') as f:
		for line in f:
			stop_words.append(line)
stop_words.extend(['???', '???', '???', '???', '????', '???', '?', '?', '??'])

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
    if token=="??" or ((len(token)!=1 and token[0].isalnum()) and (not ({'NPRO'} in wordForm.tag or{'CONJ'} in wordForm.tag or {'PREP'} in wordForm.tag or {'PRED'} in wordForm.tag or {'PRCL'} in wordForm.tag or {'INTJ'} in wordForm.tag ))):
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
######## 

prettyPrint("VEctorizing")

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    stop_words=stop_words,
	binary = True,
    max_features = 1300,
    encoding='utf-8',
	ngram_range=(1,1)
)

prettyPrint("Changing panda")
for i in range(0,len(train_data_df)):
	train_data_df.Text[i]=" ".join(tokenize(train_data_df.Text[i]))

corpus_data_features = vectorizer.fit_transform(train_data_df.Text.tolist()).toarray()
test_features = vectorizer.transform(train_data_df.Text.tolist())
#corpus_data_features = vectorizer.fit_transform(train_data_df.Text.tolist() +test_data_df.Text.tolist()).toarray()
#test = vectorizer.transform(test_data_df).toarray()
X_train, X_test, y_train, y_test  = train_test_split(
        corpus_data_features[0:len(train_data_df)], 
        train_data_df.Sentiment,
        train_size=0.80, 
        random_state=1234)

nb = naive_bayes.MultinomialNB()
cv_score = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print(classification_report(y_test, y_pred))
vocab = vectorizer.get_feature_names()
with io.open("FeatureNames.txt",'w',encoding='utf8') as f:
        print("writing")
        for w in vocab:
            f.write(w+"\r\n")