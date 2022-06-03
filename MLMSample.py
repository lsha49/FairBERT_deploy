import pandas as pd
import numpy as np
import time
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections.abc import Iterable
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from nltk import tokenize
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import json
import nltk
import textstat
import re
from random import randrange
from nltk.corpus import wordnet


FileName = '../../uq67_scratch/bfiledata/forum_2021_lang_selected_sample.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')


# nltk.download('omw-1.4')

newCorpus = pd.DataFrame()
newIndex = 0
for index,entry in enumerate(Corpus['forum_message']):
    sents = sent_tokenize(entry)
    
    for sent in sents: 
        sent = sent.replace("   ", "")
        sent = re.sub("[\(\[].*?[\)\]]", "", sent)
        words = word_tokenize(sent)
        # print(words);exit()
        # masked = False
        if ':' in words:
            continue; 

        masked = False 
        newWordsMasked = ''
        newWordsOrig = ''
        for word in words:
            if word.isalpha() and len(word) > 3 and randrange(10) == 1 and masked == False and wordnet.synsets(word):
                toWrite = ' [MASK]'  
                toWriteOriginal = ' ' + word
                masked = True 
            elif len(word) > 20: 
                 toWrite = ''
                 toWriteOriginal = ''
            else:
                toWrite = ' ' + word
                toWriteOriginal = ' ' + word
            newWordsMasked = newWordsMasked + toWrite
            newWordsOrig = newWordsOrig + toWriteOriginal
        
        newCorpus.loc[newIndex,'gender'] = Corpus.loc[index,'gender']
        newCorpus.loc[newIndex,'home_language'] = Corpus.loc[index,'home_language']
        newCorpus.loc[newIndex,'indexx'] = Corpus.loc[index,'indexx']
        
        newCorpus.loc[newIndex,'masked'] = newWordsMasked
        newCorpus.loc[newIndex,'original'] = newWordsOrig
        
        newIndex = newIndex + 1

for index,entry in enumerate(newCorpus['masked']):
    if len(str(newCorpus.loc[index, 'masked']).strip()) < 50:  
        newCorpus.loc[index, 'masked'] = ''
    countAlpha=0
    for i in str(newCorpus.loc[index, 'masked']).strip():
        if(i.isalpha()):
            countAlpha=countAlpha+1
    if countAlpha < 20:
        newCorpus.loc[index, 'masked'] = ''
    if '[MASK]' not in str(entry):  
        newCorpus.loc[index, 'masked'] = ''
    if 'George' in str(entry):  
        newCorpus.loc[index, 'masked'] = ''
    if 'george' in str(entry):  
        newCorpus.loc[index, 'masked'] = ''
    if entry.find('[MASK]') == 0:
        newCorpus.loc[index, 'masked'] = ''
    if entry.find('[MASK]') == 1:
        newCorpus.loc[index, 'masked'] = ''
    if entry.find('[MASK]') == 2:
        newCorpus.loc[index, 'masked'] = ''
    

newCorpus['masked'].replace('', np.nan, inplace=True)
newCorpus.dropna(subset=['masked'], inplace=True)

newCorpus.to_csv('../../uq67_scratch/bfiledata/forum_2021_lang_selected_sample.csv',index=False)
    


