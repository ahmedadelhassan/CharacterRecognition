'''
Created on Nov 13, 2015
@author: ahmad
'''
from DatasetReader import DatasetReader
from sklearn import naive_bayes, cross_validation
import numpy
import os
from optparse import isbasestring

class NaiveBayes(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.dsr = DatasetReader()
        self.GNB = naive_bayes.GaussianNB()
        
        
    def GaussianNB_train(self, dataset_path, cv=1):
        dataset = self.dsr.read_dataset_images(dataset_path)
        _, images, labels = self.dsr.gen_labelled_arrays(dataset)
        images = numpy.array(images)
        #reshape images for input
        data = images.reshape(len(images), -1)
        if cv <= 1:
            self.GNB.fit(data, labels)
        elif cv > 1:
            cv_result = cross_validation.cross_val_score(self.GNB, data, labels, cv=cv)
            print cv_result
        
    def GaussianNB_predict_one(self, image_path):
        if isbasestring(image_path):
            image = self.dsr.read_img_bw(image_path)
        else:
            image = image_path
        image = image.reshape(-1, image.shape[0]*image.shape[1])
        result = self.GNB.predict(image)
        
        return result
        
# from NaiveBayes import NaiveBayes
# NB = NaiveBayes()
# NB.GaussianNB_train('I:\\eclipse_workspace\\CharacterRecognition\\digits_dataset_clean', cv=5)
# print NB.GaussianNB_predict_one('I:\\eclipse_workspace\\CharacterRecognition\\test1.jpg')