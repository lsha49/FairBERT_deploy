from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments, BertTokenizer
from transformers import Trainer, AutoModelForSequenceClassification, BertForSequenceClassification
import pandas as pd
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
import torch
import tensorflow as tf
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

tokenizer = AutoTokenizer.from_pretrained(xxx,model_max_length=512)
model = AutoModelForSequenceClassification.from_pretrained("../../xxx", output_hidden_states=True)

Corpus = pd.read_csv('xxx', encoding='latin-1')

for index,entry in enumerate(Corpus['xxx']):
    input_ids = torch.tensor(tokenizer.encode(entry,truncation=True)).unsqueeze(0)  
    outputs = model(input_ids)

    hidden_states = outputs[1]
    embedding_output = hidden_states[12].detach().numpy()[0]
    finalEmb = embedding_output[len(embedding_output)-1]

    for iindex,ientry in enumerate(finalEmb):
        Corpus.loc[index, iindex] = str(ientry)

Corpus.to_csv('../../xxx',index=False)
