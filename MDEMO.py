from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
import torch
from sklearn import model_selection, naive_bayes, svm
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


# Corpus = pd.read_csv('../../uq67_scratch/bfiledata/Monash_fine_tune_test_embed.csv', encoding='latin-1')

Corpus = pd.read_csv('../../uq67_scratch/bfiledata_steve/embedding_file/yh_gender_conf_30_embedding.csv', encoding='latin-1')

# using gender language  
labelCol = np.where(Corpus['gender']=='F', 0, 1)
# labelCol = np.where(Corpus['home_language'].str.contains('english', case=False), 1, 0) # native is 1

Corpus.drop('gender', inplace=True, axis=1)
Corpus.drop('home_language', inplace=True, axis=1)
Corpus.drop('label', inplace=True, axis=1)
Corpus.drop('forum_message', inplace=True, axis=1)
Corpus = Corpus.replace(np.nan, 0)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus, labelCol, test_size=0.2, random_state=5342)

### kmeans
# kmeans = KMeans(n_clusters=2, random_state=0).fit(Train_X)
# predicted = kmeans.predict(Test_X)
# predicted = np.logical_not(predicted).astype(int)  #reverse labels if required

### Logistic regression
lr_clf=LogisticRegression()
lr_clf.fit(Train_X,Train_Y)
predicted = lr_clf.predict(Test_X)
preditedProb = lr_clf.predict_proba(Test_X)
preditedProb1 = preditedProb[:, 1]


print(predicted)

print("Accuracy Score -> ",accuracy_score(predicted, Test_Y))
print("Kappa Score -> ",cohen_kappa_score(predicted, Test_Y))
# print("AUC Score -> ", roc_auc_score(Test_Y,predicted))
print("AUC Score -> ", roc_auc_score(Test_Y,preditedProb1))
print("F1 Score -> ",f1_score(predicted, Test_Y, average='weighted'))
