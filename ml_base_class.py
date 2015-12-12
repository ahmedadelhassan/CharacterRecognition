# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:54:05 2015

@author: osm3000
"""

from DatasetReader import DatasetReader
from edf import *
from FreemanEncoder import *
import numpy as np
#from sklearn import metrics
#from sklearn import datasets, neighbors, linear_model
from sklearn import cross_validation
#import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from copy import deepcopy
import random
class ml_alg_base():
    def __init__(self):
        self.reader = DatasetReader()
    
    def training():
        print "Training method is not implemented"
        raise NotImplementedError
        
    def predict(self):
        print "Predicting method is not implemented"
        raise NotImplementedError
        
    def shuffle_data(self, data_x, data_y):
        """
        The code for this part is taken from
        http://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
        """
        c = list(zip(data_x, data_y))
        random.shuffle(c)
        data_x, data_y = zip(*c)
        return list(data_x), list(data_y)
        
    def CrossValidationScore(self, data_x, data_y, estimator):
        pass
    
    def TrainingScore(self, data_x, data_y, estimator):
        pass
    
    def TestScore(self, data_x, data_y, estimator):
        pass
    
    def first_exp(self, data_x, data_y, estimator, num_iter=10):
        local_data_x = deepcopy(data_x)
        local_data_y = deepcopy(data_y)
        
        for i in range(num_iter):
            local_data_x, local_data_y = self.shuffle_data(local_data_x, local_data_y)
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(local_data_x, local_data_y, test_size=0.2)
            
            estimator.fit(X_train, y_train)            
            
            #I will change this to leave one out cross validation
            cv_scores = cross_validation.cross_val_score(estimator, X_train, y_train, cv=10).mean()
            
            #Train score
            train_score = estimator.score(X_train, y_train)
            
            #Test score
            test_score = estimator.score(X_test, y_test)
            test_predict = estimator.predict(X_test)
            
            print("CV_Accuracy: %0.2f" % (cv_scores))
            print("train_Accuracy: %0.2f" % (train_score))
            print("test_Accuracy: %0.2f" % (test_score))
            
            print [i for i, j in zip(list(test_predict), y_test) if i != j]
            
            print "----------------------------"