from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np
from datasets import load_metric
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


bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForPreTraining.from_pretrained("bert-base-uncased")

dataset = TextDatasetForNextSentencePrediction(
    tokenizer=bert_tokenizer,
    file_path="xxx",
    block_size = xxx
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=bert_tokenizer, 
    mlm=True,
    mlm_probability= xxx
)

training_args = TrainingArguments(
    output_dir= "xxx",
    overwrite_output_dir=True,
    num_train_epochs=xxx,
    per_gpu_train_batch_size=xxx,
    save_steps=xxx,
    save_total_limit=xxx,
    prediction_loss_only=True,
    learning_rate=xxx,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model()