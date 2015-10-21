'''
Created on Oct 19, 2015

@author: ahmad
'''
import nltk
from DatasetReader import DatasetReader
from FreemanEncoder import FreemanEncoder

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
        
    def generate_labelled_sequences(self, freeman_codes_dict):
        labelled_sequences = []
        codes_list = freeman_codes_dict.items()
        for tup in codes_list:
            for code in tup[1]:
                labelled_sequences.append((tup[0],code))
                
        return labelled_sequences
            
        
    def hmm_train(self, dataset_path):
        dataset = self.dsr.read_dataset_images(dataset_path)
        freeman_codes_dict = self.fenc.encode_freeman_dataset(dataset)
         
        labelled_sequences = self.generate_labelled_sequences(freeman_codes_dict)
        
        hmm_inst = nltk.tag.hmm.HiddenMarkovModelTrainer
        trained_hmm = hmm_inst.train_supervised(labelled_sequences)
        
        return trained_hmm

## TESTING CODE (WILL BE REMOVED) ##
from HMM import HMM
hmm = HMM()
thmm = hmm.hmm_train('I:\\eclipse_workspace\\CharacterRecognition\\digits_dataset')