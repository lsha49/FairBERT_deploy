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
from alipy import ToolBox
from collections.abc import Iterable
from alipy.query_strategy import QueryInstanceLAL,QueryInstanceQUIRE,QueryInstanceSPAL
from imblearn.under_sampling import RandomUnderSampler
from deslib.util.instance_hardness import kdn_score
from scipy.spatial import distance

class Selector(object):
    def AlSamples(self,X,y,labelledSet,unLabelledSet): 
        alibox = ToolBox(X=X, y=y)
        # QueryInstanceQBC
        # QueryInstanceUncertainty
        # QueryExpectedErrorReduction
        # QueryInstanceLAL
        # ...etc.,
        Strategy = alibox.get_query_strategy(strategy_name='XXX')
        select_ind = Strategy.select(labelledSet, unLabelledSet, model=xxx, batch_size=xxx)
