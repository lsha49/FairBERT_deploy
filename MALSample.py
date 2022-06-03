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



# forum_2021_lang_train_embed_bert_base
# forum_2021_lang_unselected_sample
Corpus = pd.read_csv('../../uq67_scratch/bfiledata/forum_2021_lang_train_embed_bert_base.csv', encoding='latin-1')
selectSamplesGroup1 = 30000
selectSamplesGroup0 = 5000


FineTuneCorpus = pd.read_csv('../../uq67_scratch/bfiledata/Monash_fine_tune_embed.csv', encoding='latin-1')

labelFineY = np.where(pd.isnull(FineTuneCorpus['label']), 0, 1)

# labelFineG = np.where(FineTuneCorpus['lang'].str.contains('english', case=False), 1, 0) # native is 1
# labelG = np.where(Corpus['home_language'].str.contains('english', case=False), 1, 0) # native is 1

labelFineG = np.where(FineTuneCorpus['gender'] == 'F', 0, 1) 
labelG = np.where(Corpus['gender'] == 'F', 0, 1) 


# get BERT embeddings fine tune samples
FineTuneCorpus.drop('facaulty', inplace=True, axis=1)
FineTuneCorpus.drop('label', inplace=True, axis=1)
FineTuneCorpus.drop('gender', inplace=True, axis=1)
FineTuneCorpus.drop('lang', inplace=True, axis=1)
featuresFine = FineTuneCorpus.replace(np.nan, 0)

originalCorpus = Corpus.copy()

# get BERT embeddings pre-train samples
Corpus.drop('gender', inplace=True, axis=1)
Corpus.drop('home_language', inplace=True, axis=1)
Corpus.drop('birth_country', inplace=True, axis=1)
Corpus.drop('indexx', inplace=True, axis=1)
Corpus.drop('person_id', inplace=True, axis=1)
Corpus.drop('forum_message', inplace=True, axis=1)
features = Corpus.replace(np.nan, 0)


###### select a small fair task-labelled set for AL-based selection ######
labelFineYG = np.char.add(labelFineY.astype(str), labelFineG.astype(str))
taskInd00 = np.where(labelFineYG=='00')[0]
taskInd01 = np.where(labelFineYG=='01')[0]
taskInd10 = np.where(labelFineYG=='10')[0]
taskInd11 = np.where(labelFineYG=='11')[0]
taskInd00_ran = np.random.choice(taskInd00, size=250, replace=False)
taskInd01_ran = np.random.choice(taskInd01, size=250, replace=False)
taskInd10_ran = np.random.choice(taskInd10, size=250, replace=False)
taskInd11_ran = np.random.choice(taskInd11, size=250, replace=False)
taskAll = np.concatenate([taskInd00,taskInd01,taskInd10,taskInd11])
tasklabelledIndList = np.concatenate([taskInd00_ran,taskInd01_ran,taskInd10_ran,taskInd11_ran])

selectedFineTuneSetFeatures = featuresFine.loc[tasklabelledIndList]
selectedFineTuneSetLabelFineY = labelFineY[tasklabelledIndList]
selectedFineTuneSetLabelFineG = labelFineG[tasklabelledIndList]

# print(features);exit()

allSample = np.concatenate([selectedFineTuneSetFeatures,features])
allLabelT = np.concatenate([selectedFineTuneSetLabelFineY,labelG])
allLabelG = np.concatenate([selectedFineTuneSetLabelFineG,labelG])
# kdnResult = kdn_score(featuresFine, labelFineY, 5)
# KDNlist00 = kdnResult[0][taskInd00_ran]
# KDNlist01 = kdnResult[0][taskInd01_ran]
# KDNlist10 = kdnResult[0][taskInd10_ran]
# KDNlist11 = kdnResult[0][taskInd11_ran]
# kl_pq0 = distance.jensenshannon(KDNlist00, KDNlist01)
# kl_pq1 = distance.jensenshannon(KDNlist10, KDNlist11)
# print('H-bias:', (kl_pq0 + kl_pq1)/2)
# savedTasklabelledIndList = pd.DataFrame(tasklabelledIndList)
# savedTasklabelledIndList.to_csv('savedTasklabelledIndList.csv',index=False)


labelledSet = []
unLabelledSet = []
for i in range(0, 1000):
    labelledSet = labelledSet + [i]

for i in range(1001, len(allLabelG)-1):
    unLabelledSet = unLabelledSet + [i]

corpusIndices = []
for i in range(len(Corpus)):
    corpusIndices.append(i)

###### AL select samples ######

### QueryInstanceQBC: query-by-committee, fast
# alibox = ToolBox(X=allSample, y=allLabelT) # select task-informative samples
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceQBC')
# select_ind_task = Strategy.select(labelledSet, unLabelledSet, model=None, batch_size=100000)
# alibox = ToolBox(X=allSample, y=allLabelG) # select demo-uninformative samples
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceQBC')
# select_ind_demo_un = Strategy.select(labelledSet, unLabelledSet, model=None, batch_size=100000)


### QueryInstanceUncertainty: uncertainity, fast
alibox = ToolBox(X=allSample, y=allLabelT, measure='least_confident')
Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceUncertainty')
select_ind_task = Strategy.select(labelledSet, unLabelledSet, model=None, batch_size=100000)
alibox = ToolBox(X=allSample, y=allLabelG, measure='least_confident')
Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceUncertainty')
select_ind_demo_un = Strategy.select(labelledSet, unLabelledSet, model=None, batch_size=100000)


### QueryExpectedErrorReduction: Expected Error reduction ### this is taking more than a day
# alibox = ToolBox(X=allSample, y=allLabelT)
# Strategy = alibox.get_query_strategy(strategy_name='QueryExpectedErrorReduction')
# select_ind_task = Strategy.select(labelledSet, unLabelledSet, model=None, batch_size=10000)
# print('selection1 finished')
# alibox = ToolBox(X=allSample, y=allLabelG)
# Strategy = alibox.get_query_strategy(strategy_name='QueryExpectedErrorReduction')
# select_ind_demo_un = Strategy.select(labelledSet, unLabelledSet, model=None, batch_size=10000)
# print('selection2 finished')


### QueryExpectedErrorReduction: LAL EER
# alibox = ToolBox(X=allSample, y=allLabelT,query_type='AllLabels', mode='LAL_independent',train_slt=True)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceLAL')
# select_ind_task = Strategy.select(labelledSet, unLabelledSet, model=None, batch_size=100000)
# alibox = ToolBox(X=allSample, y=allLabelG,query_type='AllLabels', mode='LAL_independent',train_slt=True)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceLAL')
# select_ind_demo_un = Strategy.select(labelledSet, unLabelledSet, model=None, batch_size=100000)



# intercept of task and demo samples of 10000
select_ind_demo = list(set(corpusIndices) - set(select_ind_demo_un))
selected_ind = np.intersect1d(select_ind_demo, select_ind_task); print(len(selected_ind))

demo1index = np.where(labelG==1)[0]
demo0index = np.where(labelG==0)[0]

selected_ind_demo1 = np.intersect1d(selected_ind, demo1index)[:selectSamplesGroup1]
selected_ind_demo0 = np.intersect1d(selected_ind, demo0index)[:selectSamplesGroup0]
selected_ind = np.concatenate([selected_ind_demo1,selected_ind_demo0])

# selected_ind = selected_ind[:20000]

un_select_ind = list(set(corpusIndices) - set(selected_ind))

### QueryExpectedErrorReduction: Expected Error reduction
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryExpectedErrorReduction')
# select_ind = Strategy.select(firstIndList, secondIndList, model=None, batch_size=100)

### QueryInstanceGraphDensity: representativeness
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceGraphDensity', train_idx=allIndList)
# select_ind = Strategy.select(firstIndList, allIndList, batch_size=100)

### QueryInstanceLAL: Expected Error Reduction on a trained regressor
# alibox = ToolBox(X=features, y=label, query_type='AllLabels', saving_path='')
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceLAL')
# select_ind = Strategy.select(firstIndList, secondIndList, model=None, batch_size=100)

### QueryInstanceBMDR, representative and informative
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceBMDR')
# select_ind = Strategy.select(firstIndList, secondIndList, batch_size=100)


selectedCorpus = originalCorpus.loc[selected_ind]

unselectedCorpus = originalCorpus.loc[un_select_ind] 

selectedCorpus.to_csv('../../uq67_scratch/bfiledata/forum_2021_lang_selected_sample.csv',index=False)

unselectedCorpus.to_csv('../../uq67_scratch/bfiledata/forum_2021_lang_unselected_sample.csv',index=False)
