from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np
# from datasets import load_metric
import torch
import tensorflow as tf
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertForMaskedLM, BertForPreTraining, BertConfig
from transformers import TextDatasetForNextSentencePrediction
from transformers import DataCollatorForLanguageModeling


Corpus = pd.read_csv('data/XXX.csv', encoding='latin-1')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

inputs = tokenizer(Corpus['masked'].tolist(), return_tensors="pt", truncation=True, padding=True, max_length=xxx)
inputs['labels'] = tokenizer(Corpus['original'].tolist(), return_tensors="pt",  truncation=True, padding=True, max_length=xxx)["input_ids"]


args = TrainingArguments(
    overwrite_output_dir=True,
    output_dir='saved_model/xxx',
    per_device_train_batch_size=xx,
    num_train_epochs=xx,
    learning_rate=xx,
    save_strategy='epoch',
)

class EncodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = EncodeDataset(inputs)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()
trainer.save_model()
