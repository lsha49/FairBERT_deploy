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

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForSequenceClassification.from_pretrained("saved_model/further_1/checkpoint-6500", output_hidden_states=True, local_files_only=True)
# model = BertForSequenceClassification.from_pretrained("saved_model/further_1/checkpoint-6500", output_hidden_states=True)
# model = BertForSequenceClassification.from_pretrained("bert-base-cased", output_hidden_states=True)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased",model_max_length=512)

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", output_hidden_states=True)

model = AutoModelForSequenceClassification.from_pretrained("../../uq67_scratch/bfiledata/lele_test_lang", output_hidden_states=True)

# model = AutoModelForSequenceClassification.from_pretrained("../../uq67_scratch/saved_model/100checkpoint/checkpoint-46000", output_hidden_states=True)


# Monash_fine_tune
# Monash_fine_tune_test_incre
Corpus = pd.read_csv('Monash_fine_tune.csv', encoding='latin-1')


excep = 0
for index,entry in enumerate(Corpus['forum_message']):
    input_ids = torch.tensor(tokenizer.encode(entry,truncation=True)).unsqueeze(0)  
    outputs = model(input_ids)

    hidden_states = outputs[1]
    embedding_output = hidden_states[12].detach().numpy()[0]
    finalEmb = embedding_output[len(embedding_output)-1]

    for iindex,ientry in enumerate(finalEmb):
        Corpus.loc[index, iindex] = str(ientry)

# Monash_fine_tune_clean_embed
Corpus.to_csv('../../uq67_scratch/bfiledata/Monash_fine_tune_test_embed.csv',index=False)
