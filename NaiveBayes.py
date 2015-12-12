'''
Created on Nov 13, 2015
@author: ahmad
'''
from DatasetReader import DatasetReader
from sklearn import naive_bayes, cross_validation
import numpy
import os
from optparse import isbasestring
import pickle
from ml_base_class import ml_alg_base

class NaiveBayes(ml_alg_base):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        ml_alg_base.__init__(self)
        self.dsr = DatasetReader()
        self.learning_model = naive_bayes.GaussianNB()
        
    def get_data(self, dataset_path = "./teams_dataset"):
        data_dict = self.dsr.read_dataset_images(dataset_path)
        
        _,data_set_x, data_set_y = self.dsr.gen_labelled_arrays(data_dict)
        data_set_x = data_set_x.reshape(len(data_set_x), -1)
        
        return data_set_x, data_set_y
        
    def training(self, dataset_path, cv=1):
        dataset = self.dsr.read_dataset_images(dataset_path)
        _, images, labels = self.dsr.gen_labelled_arrays(dataset)
        images = numpy.array(images)
        #reshape images for input
        data = images.reshape(len(images), -1)
        if cv <= 1:
            self.learning_model.fit(data, labels)
        elif cv > 1:
            cv_result = cross_validation.cross_val_score(self.learning_model, data, labels, cv=cv)
            return cv_result
            
        pickle.dump( self.learning_model, open( "./Models/naivebayes_model.p", "wb" ) )
        
    def predict(self, image_path):
        try:
            self.learning_model = pickle.load( open( "./Models/naivebayes_model.p", "rb" ) )
        except:
            print "Please train the Naive Bayes model first"
        
        if isbasestring(image_path):
            image = self.dsr.read_img_bw(image_path)
        else:
            image = image_path
        image = image.reshape(-1, image.shape[0]*image.shape[1])
        result = self.learning_model.predict(image)
        
        return result
        
# from NaiveBayes import NaiveBayes
# NB = NaiveBayes()
# # NB.training('I:\\eclipse_workspace\\CharacterRecognition\\digits_dataset_clean', cv=5)
# # print NB.predict('I:\\eclipse_workspace\\CharacterRecognition\\test1.jpg')
# data_x, data_y = NB.get_data()
# print data_x.shape, data_y.shape
# NB.first_exp(data_x, data_y, NB.learning_model, algorithm_name='NaiveBayes' ,num_iter=50)