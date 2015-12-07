from DatasetReader import DatasetReader
from edf import elliptic_fourier_descriptors
import numpy as np
#from sklearn import metrics
#from skl1earn.cross_validation import train_test_split
#from sklearn import datasets, neighbors, linear_model
#from sklearn import cross_validation
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB

class RandomForestsasdas():
    def __init__(self, num_fourier_des = 10):
        self.reader = DatasetReader()
        self.num_fourier_des = num_fourier_des
        self.random_forest = RandomForestClassifier(max_depth=10, n_estimators=10)
    
    def training(self, data_set_path = "./digits_dataset", cv=1):
        data_dict = self.reader.read_dataset_images(data_set_path)
        _,data_set_x, data_set_y = self.reader.gen_labelled_arrays(data_dict)

        training_data = []
        for image_array in data_set_x:
            efds1, K1, T1 = elliptic_fourier_descriptors(image_array,self.num_fourier_des)
            training_data.append(np.reshape(efds1[0], (1,-1))[0])
            
        self.random_forest.fit(training_data, data_set_y)
        
    def predict(self, image):
        efds1, K1, T1 = elliptic_fourier_descriptors(image,self.num_fourier_des)
        test_data = np.reshape(efds1[0], (1,-1))[0]
        
        return self.random_forest.predict(test_data)
    
    def predict_debug(self, image='./teams_dataset/9/9_0.jpg'):
        image = self.reader.read_img_bw(image)
        efds1, K1, T1 = elliptic_fourier_descriptors(image,self.num_fourier_des)
        test_data = np.reshape(efds1[0], (1,-1))[0]
                
        return self.random_forest.predict(test_data)
    
# model = RandomForests()
# model.training()
# model.predict_debug()