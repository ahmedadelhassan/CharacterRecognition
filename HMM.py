'''
Created on Oct 19, 2015

@author: ahmad
'''

import numpy
import os
from DatasetReader import DatasetReader
from FreemanEncoder import FreemanEncoder
from sklearn import cross_validation
from nltk import HiddenMarkovModelTrainer
import pickle

class HMM(object):
    '''
    classdocs
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        self.dsr = DatasetReader()
        self.fenc = FreemanEncoder()
        self.learning_model = HiddenMarkovModelTrainer()
        self.model = None
        
    def generate_labelled_sequences(self, freeman_codes_dict):
        labeled_sequences = []
        labeled_symbols = []
        codes_list = freeman_codes_dict.items()
        for tup in codes_list:
            for code in tup[1]:
                temp = []
                for symbol in code:
                    temp.append((symbol, tup[0]))
                labeled_symbols.append(temp)
        for tup in codes_list:
            for code in tup[1]:
                labeled_sequences.append((code,tup[0]))
                
        codes = numpy.array([x[0] for x in labeled_sequences])
        labels = numpy.array([y[1] for y in labeled_sequences])
        return labeled_symbols, labeled_sequences, codes, labels
            
        
    def training(self, dataset_path, cv=1, n_iter=1):
        dataset = self.dsr.read_dataset_images(dataset_path)
        freeman_codes_dict = self.fenc.encode_freeman_dataset(dataset)
         
        labeled_symbols, labeled_sequence, codes, labels = self.generate_labelled_sequences(freeman_codes_dict)
        
        self.model = self.learning_model.train(labeled_symbols)
        
        if cv > 1:
            cv_scores = []
            for i in range(n_iter):
                skf = cross_validation.KFold(len(labels), n_folds=10, shuffle=True)
                
                iter_score = []
                for train_index, test_index in skf:
                    train_data = list(numpy.array(labeled_symbols)[train_index])
                    test_data = list(numpy.array(labeled_symbols)[test_index])
                    self.model = self.learning_model.train(train_data)
                    fold_score = self.model.evaluate(test_data)
                    iter_score.append(fold_score)
                
                cv_scores.append(numpy.mean(iter_score))
                
            return cv_scores
        else:
            skf = cross_validation.ShuffleSplit(len(labels), n_iter=n_iter, test_size=0.2, random_state=0)
            training_score = []
            test_score = []
            for train_index, test_index in skf:
                train_data = list(numpy.array(labeled_symbols)[train_index])
                test_data = list(numpy.array(labeled_symbols)[test_index])
                self.model = self.learning_model.train(train_data)
                training_score.append(self.model.evaluate(train_data))
                test_score.append(self.model.evaluate(test_data))
                
                
            return training_score, test_score
    
    def predict(self, image_path):
        
        if os.path.isfile(image_path):
            image_array = self.dsr.read_img_bw(image_path)
            freeman_code = self.fenc.encode_freeman(image_array)
        else:
            freeman_code = image_path
        
        
        predicted_states = self.model.tag(freeman_code)
        predicted_states = [x[1] for x in predicted_states]
        if len(set(predicted_states)) == 1:
            predicted_class = list(set(predicted_states))[0]
        
        return predicted_class
        

## TESTING CODE (WILL BE REMOVED) ##
# from HMM import HMM
# hmm = HMM()
# cv_scores = hmm.training('I:\\eclipse_workspace\\CharacterRecognition\\teams_dataset', cv=10, n_iter=50)
# train_score, test_score = hmm.training('I:\\eclipse_workspace\\CharacterRecognition\\teams_dataset', n_iter=50)
#  
# with open("./Results/hmm.txt", 'w') as fp:
#     for i in range(len(cv_scores)):
#         text = str(cv_scores[i]) + ',' + str(train_score[i]) + ',' + str(test_score[i]) + '\n'
#         print text
#         print '--------------------------------'
#         fp.write(text)