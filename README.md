# FairBERT: Fair pretraining of BERT for Fair Modeling in Education
An python repository to perform educational dataset balancing applied in submitted paper in @todo. 

Download this repository with `git clone` or equivalent.

```bash
git clone @repo
```

## Requirements  
* Python 3.8  
* Tensorflow > 1.5
* tensorflow-estimator 2.7.0
* tensorflow-macos 2.7.0
* tensorflow-metal 0.3.0
* Sklearn > 0.19.0  


## Seed Dataset, Hardness Bias and Fairness implementation 
We detail below how to implement hard-bias and fairness evaluation. See example code in ```Util.py```

### Hardness Bias
After generating samples, we evaluate the kDN distribution by ``` distance.jensenshannon ``` and selected the lowest H-bias samples. The H-bias can be calculated by ``` calKDN ``` function.


### Fairness Evaluation 
We applied ``` abroca ``` package in [ABROCA](https://pypi.org/project/abroca/). A sample calculation of ABROCA: 
```
slice = compute_abroca(abrocaDf, 
        pred_col = 'prob_1' , 
        label_col = 'label', 
        protected_attr_col = 'gender',
        majority_protected_attr_val = '2',
        compare_type = 'binary', # binary, overall, etc...
        n_grid = 10000,
        plot_slices = False)
```


## Model implementation detail
A Logistic regression model is implemented in ```Util.py``` by ``` logisticRegression ``` function. 

```
A sample GridSearched model: 
lrc = LogisticRegression(C=4.281332398719396, class_weight=None, dual=False,
    fit_intercept=True, intercept_scaling=1, max_iter=100,
    n_jobs=1, penalty='l1', random_state=None,
    solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
```

## Embedding extraction implementation detail
A sample embedding extraction from BERT model is implemented in ```MEmb.py``


## Further pretraining implementation detail
We followed the same pretraining procedule as shown in [huggingface](https://huggingface.co/docs/transformers/model_doc/bert#overview)
See a sample implmentation in ```MTrain```.


## AL sampling implementation detail
AL sampling is implemented by [alipy](http://parnec.nuaa.edu.cn/huangsj/alipy/)
See a sample implmentation in ```MALSample```.



