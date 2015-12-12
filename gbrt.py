from DatasetReader import DatasetReader
from edf import *
from FreemanEncoder import *
import numpy as np
#from sklearn import metrics
from sklearn.cross_validation import train_test_split
#from sklearn import datasets, neighbors, linear_model
from sklearn import cross_validation
#import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import random
from ml_base_class import ml_alg_base
import pickle
from sklearn.ensemble import GradientBoostingClassifier

class GBRT(ml_alg_base):
    def __init__(self, num_fourier_des = 10):
        ml_alg_base.__init__(self)
        self.num_fourier_des = num_fourier_des

        self.learning_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=5)
    
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
        
        pickle.dump( self.learning_model, open( "./Models/GradientBoostingClassifier.p", "wb" ) )
        
    def predict(self, image):
        try:
            self.learning_model = pickle.load( open( "./Models/GradientBoostingClassifier.p", "rb" ) )
        except:
            print "Please train the logistic model first"
            #exit()
        fourier_desc = self.get_fourier_desc(image)
        test_data = np.reshape(fourier_desc, (1,-1))[0]
        predictions = self.learning_model.predict(test_data)
        return map(str, predictions) # I return str, since I am not sure ADEL is working with integers
    
    def get_fourier_desc(self, image_array):
        efds1, K1, T1 = elliptic_fourier_descriptors(image_array,self.num_fourier_des)
        return efds1[0]
        
# The grid search code - to find the best parameters
#classifier = GBRT()
#data_x, data_y = classifier.get_data()
#classifier.first_exp(data_x, data_y, classifier.learning_model, num_iter=50, algorithm_name="gbrt") #change 10 later to 50