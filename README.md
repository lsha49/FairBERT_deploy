# FAIR BERT PROJECT
This is a casenote script file containing model training and testing using LEGAL BERT model published in huggingface

## Requirements  
* The project is developed using the latest version of tensorflow (wheel is for Mac M1)
* The most critical dependency is HuggingFace, python3

## Interface for input 
During development, the only input file needed is a csv file containing: 
* col1: Sentence 
* col2-n: the six label categories of the sentence 


## How to run this? 
First add a data.csv file under data/ folder, then a test run is as simple as 
```
python3 tfBertClf.py
```


