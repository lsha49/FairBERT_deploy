
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import math
import imblearn
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from deslib.util.instance_hardness import kdn_score
from numpy import where
import random
from math import log2, log, sqrt
from abroca import *
from scipy.spatial import distance
from statistics import stdev, mean

class Util(object):
    def preprocessing(self, filename = "dataFile", useLabel = "Label"): 
        """import and split dataset for data balancing experiment.
        Parameters
        ----------
        filename: string, optional (default='dataFile')
            applied to import your data file
        useLabel: string, optional (default='Label')
            label column name in the csv file 
            e.g. is_content_related
        Returns
        -------
        """
        # @todo import data and preprocessing
        Corpus = pd.read_csv(filename, encoding='') 


    def calKDN(self, features, labels,  List00Index, List01Index, List10Index, List11Index): 
        """calculate Hardness-bias by KDN distribution.
        Parameters
        ----------
        features: array-like,
            Feature matrix with [n_samples, n_features].
        labels: array-like,
            Labels of given data [n_samples]
        List00Index: array-like,
            indices of (protected attribute=0, label=0) in the feature and label set
        List01Index: array-like,
            indices of (protected attribute=0, label=1) in the feature and label set
        List10Index: array-like,
            indices of (protected attribute=1, label=0) in the feature and label set
        List11Index: array-like,
            indices of (protected attribute=1, label=1) in the feature and label set
        Returns
        -------
        avg_kl_pq_01: float,
            Hardness bias score.
        """
        kdnResult = kdn_score(features, labels, 5)

        kdn_counts_00 = dict(Counter(kdnResult[0][List00Index]))
        kdn_counts_01 = dict(Counter(kdnResult[0][List01Index]))
        kdn_counts_10 = dict(Counter(kdnResult[0][List10Index]))
        kdn_counts_11 = dict(Counter(kdnResult[0][List11Index]))
        kdn_distribution_00 = []
        kdn_distribution_01 = []
        kdn_distribution_10 = []
        kdn_distribution_11 = []

        for i in range(6):
            if i*0.2 in kdn_counts_00.keys():
                kdn_distribution_00.append(kdn_counts_00[i*0.2]/kdnResult[0][List00Index].shape[0])
            else:
                kdn_distribution_00.append(0)
            
            if i*0.2 in kdn_counts_01.keys():
                kdn_distribution_01.append(kdn_counts_01[i*0.2]/kdnResult[0][List01Index].shape[0])
            else:
                kdn_distribution_01.append(0)
            
            if i*0.2 in kdn_counts_10.keys():
                kdn_distribution_10.append(kdn_counts_10[i*0.2]/kdnResult[0][List10Index].shape[0])
            else:
                kdn_distribution_10.append(0)
            
            if i*0.2 in kdn_counts_11.keys():
                kdn_distribution_11.append(kdn_counts_11[i*0.2]/kdnResult[0][List11Index].shape[0])
            else:
                kdn_distribution_11.append(0)
 
        kl_pq0 = distance.jensenshannon(kdn_distribution_00, kdn_distribution_10)
        kl_pq1 = distance.jensenshannon(kdn_distribution_01, kdn_distribution_11)

        return (kl_pq0 + kl_pq1)/2

    def getDfToComputeAbroca(self, predicted, predictionProb, Test_Y, Test_G):
        """prepare dataframe for ABROCA related calculations.
        Parameters
        ----------
        predicted: array-like,
            model predictions of given test data [n_samples]
        predictionProb: array-like,
            model prediction probability of given test data [n_samples, n_labels] or [n_samples]
        Test_Y: array-like,
            test set label of given test data [n_samples]
        Test_G: array-like,
            test set protected attribute of given test data [n_samples]
        Returns
        -------
        predictionDataframe: array-like,
            ABROCA dataframe containing prediction label and demographic info.
        """
        predictionDataframe = pd.DataFrame(predicted, columns = ['predicted'])
        predictionDataframe['prob_1'] = pd.DataFrame(predictionProb)[1]
        predictionDataframe['label'] = Test_Y.tolist()
        predictionDataframe['gender'] = Test_G.astype(str)
        return predictionDataframe

    def computeAbroca(self, abrocaDf):
        """perform ABROCA calculation.
        Parameters
        ----------
        abrocaDf: array-like,
            prediction dataframe built by getDfToComputeAbroca().
        Returns
        -------
        slice: array-like,
            contains abroca calculation result 
            see spec and parameter values of compute_abroca in https://pypi.org/project/abroca/
        """
        slice = compute_abroca(abrocaDf, 
                            pred_col = '' , 
                            label_col = '', 
                            protected_attr_col = '',
                            majority_protected_attr_val = '',
                            compare_type = '',
                            n_grid = 10000,
                            plot_slices = False)
        return slice
    
    def logisticRegression(self,Train_X,Train_Y,Test_X,Test_Y): 
        """sklearn LR model.
        Parameters
        ----------
        Train_X: array-like,
            feature set of training samples
        Train_Y: array-like,
            label set of training samples
        Test_X: array-like,
            feature set of testing samples
        Test_Y: array-like,
            label set of testing samples
        Returns
        -------
        """
        ### Logistic regression
        lr_clf=LogisticRegression()
        lr_clf.fit(Train_X,Train_Y)
        predicted = lr_clf.predict(Test_X)
        # preditedProb = lr_clf.predict_proba(Test_X)
        # preditedProb1 = preditedProb[:, 1]

        print("Accuracy Score -> ",accuracy_score(predicted, Test_Y))
        print("Kappa Score -> ",cohen_kappa_score(predicted, Test_Y))
        print("AUC Score -> ", roc_auc_score(Test_Y,predicted))
        # print("AUC Score -> ", roc_auc_score(Test_Y,preditedProb1))
        print("F1 Score -> ",f1_score(predicted, Test_Y, average='weighted'))

# def kfoldSplitTest(self, Corpus, labelCol):
#     kfold = KFold(5, True, 111)
#     for train_index, test_index in kfold.split(X):
#         Train_X, Test_X = Corpus[train_index], Corpus[test_index]
#         Train_Y, Test_Y = labelCol[train_index], labelCol[test_index]

# def randomSplitTest(self, Corpus, labelCol):
#     Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus, labelCol, test_size=0.2, random_state=5342)