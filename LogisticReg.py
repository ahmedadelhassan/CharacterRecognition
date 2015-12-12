from DatasetReader import DatasetReader
from edf import *
from FreemanEncoder import *
import numpy as np
#from sklearn import metrics
from sklearn.cross_validation import train_test_split
#from sklearn import datasets, neighbors, linear_model
from sklearn import cross_validation
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import random
from ml_base_class import ml_alg_base
from sklearn.externals import joblib

class LogisticReg(ml_alg_base):
    def __init__(self, num_fourier_des = 10):
        ml_alg_base.__init__(self)
        self.num_fourier_des = num_fourier_des
        """        
        The following classifier configurations has been selected by the grid search
        These were the results of running the grid search for 5 times
        {'penalty': 'l2', 'C': 10}
        0.945
        {'penalty': 'l1', 'C': 50}
        0.965
        {'penalty': 'l2', 'C': 50}
        0.955
        {'penalty': 'l2', 'C': 10}
        0.935
        {'penalty': 'l2', 'C': 50}
        0.965
        {'penalty': 'l2', 'C': 50}
        0.94
        {'penalty': 'l2', 'C': 100}
        0.965
        {'penalty': 'l2', 'C': 50}
        0.97
        {'penalty': 'l1', 'C': 10}
        0.955
        {'penalty': 'l1', 'C': 10}
        0.97
        """
        self.learning_model = LogisticRegression(multi_class='multinomial', penalty = "l2", C= 10, solver="lbfgs")
    
    def get_data(self, dataset_path = "./teams_dataset"):
        data_dict = self.reader.read_dataset_images(dataset_path)
        
        _,data_set_x, data_set_y = self.reader.gen_labelled_arrays(data_dict)
        data_set_y = map(int,data_set_y) #convert the string label into a number - may create a problem later!!
        data_set_x, data_set_y = self.shuffle_data(data_set_x, data_set_y)
        
        training_data = []
        for image_array in data_set_x:
            fourier_desc = self.get_fourier_desc(image_array)
            training_data.append(np.reshape(fourier_desc, (1,-1))[0])        
        
        return training_data, data_set_y
    
    def training(self, dataset_path = "./teams_dataset"):
        training_data, data_set_y = self.get_data(dataset_path)
            
        self.learning_model.fit(training_data, data_set_y)
        
        joblib.dump(self.learning_model, 'logistic_model.pkl') 
        
    def predict(self, image):
        try:
            self.learning_model = joblib.load('logistic_model.pkl')
        except:
            print "Please train the logistic model first"
            #exit()
        fourier_desc = self.get_fourier_desc(image)
        test_data = np.reshape(fourier_desc, (1,-1))[0]
        predictions = self.learning_model.predict(test_data)
        return map(str, predictions) # I return str, since I am not sure ADEL is working with integers
        
    def shuffle_data(self, data_x, data_y):
        """
        The code for this part is taken from
        http://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
        """
        c = list(zip(data_x, data_y))
        random.shuffle(c)
        data_x, data_y = zip(*c)
        return list(data_x), list(data_y)
    
    def get_fourier_desc(self, image_array):
        efds1, K1, T1 = elliptic_fourier_descriptors(image_array,self.num_fourier_des)
        return efds1[0]
        
    def grid_search(self):
        data_x, data_y = self.get_data()
        training_data = []
        for image_array in data_x:
            fourier_desc = self.get_fourier_desc(image_array)
            training_data.append(np.reshape(fourier_desc, (1,-1))[0])
        X_train, X_test, y_train, y_test = train_test_split(training_data, data_y, test_size=0.2, random_state=0)
        tuned_parameters = {'C':[1,10,50,100],
                            'penalty':['l1','l2']}
#        tuned_parameters = {'n_estimators':[10,20],
#                            'criterion':['gini', 'entropy'],
#                            'max_depth':[5,10]}
        clf = GridSearchCV(LogisticRegression(multi_class='multinomial'), tuned_parameters, cv=5)
        clf.fit(X_train, y_train)
        
        print clf.best_params_
        print clf.best_score_
        
# The grid search code - to find the best parameters
#classifier = LogisticReg()
#data_x, data_y = classifier.get_data()
#classifier.first_exp(data_x, data_y, classifier.learning_model, num_iter=50) #change 10 later to 50