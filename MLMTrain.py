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


Corpus = pd.read_csv('../../uq67_scratch/bfiledata/forum_2021_lang_selected_sample.csv', encoding='latin-1')

### perform BertForMaskedLM only
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# model = BertForMaskedLM.from_pretrained("../../uq67_scratch/bfiledata/lele_test_lang")

inputs = tokenizer(Corpus['masked'].tolist(), return_tensors="pt", truncation=True, padding=True, max_length=256)
inputs['labels'] = tokenizer(Corpus['original'].tolist(), return_tensors="pt",  truncation=True, padding=True, max_length=256)["input_ids"]


# lele_test_lang 
# lele_test_incre_
args = TrainingArguments(
    overwrite_output_dir=True,
    output_dir='../../uq67_scratch/bfiledata/lele_test_lang',
    per_device_train_batch_size=8,
    num_train_epochs=6,
    learning_rate=2e-5,
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
