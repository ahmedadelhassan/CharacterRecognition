'''
Created on Oct 19, 2015

@author: ahmad
'''

import numpy
import os
from seqlearn.hmm import MultinomialHMM
from DatasetReader import DatasetReader
from FreemanEncoder import FreemanEncoder
from sklearn import cross_validation
from nltk import HiddenMarkovModelTrainer

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
        self.mhmm = MultinomialHMM()
        self.nltkhmm = HiddenMarkovModelTrainer()
        self.model = ''
        
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
            
        
    def hmm_train(self, dataset_path):
        dataset = self.dsr.read_dataset_images(dataset_path)
        freeman_codes_dict = self.fenc.encode_freeman_dataset(dataset)
#         freeman_histogram = self.fenc.gen_bagofwords_dict(freeman_codes_dict)
         
        labeled_symbols, labeled_sequence, codes, labels = self.generate_labelled_sequences(freeman_codes_dict)
        
#         codes = [list(map(double,list(x))) for x in codes]
        self.model = self.nltkhmm.train(labeled_symbols)
        
        training_score = self.model.evaluate(labeled_symbols)
        
#         # Convert training data to compatible HMM structure
#         codes, indices = numpy.unique(codes, return_inverse=True)
#         X = (indices.reshape(-1, 1) == numpy.arange(len(codes))).astype(int) # -1 to infer value from original array
        
        # Train the HMM with data and labels
#         trained_hmm = self.mhmm.fit(codes, labels, [8])
        
        return training_score
    
    def hmm_predict_one(self, image_path):
        if os.path.isfile(image_path):
            image_array = self.dsr.read_img_bw(image_path)
            freeman_code = self.fenc.encode_freeman(image_array)
        else:
            freeman_code = image_path
        
#         code_histogram = self.fenc.count_bagofwords(freeman_code)
        
        predicted_states = self.model.tag(freeman_code)
        predicted_states = [x[1] for x in predicted_states]
        if len(set(predicted_states)) == 1:
            predicted_class = list(set(predicted_states))[0]
        
        return predicted_class
        

## TESTING CODE (WILL BE REMOVED) ##
# from HMM import HMM
# hmm = HMM()
# print hmm.hmm_train('I:\\eclipse_workspace\\CharacterRecognition\\digits_dataset_clean')
# print hmm.hmm_predict_one('I:\\eclipse_workspace\\CharacterRecognition\\observation\\0\\0_9.png')