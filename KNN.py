'''
Created on Oct 22, 2015

@author: tho
'''
import random, operator, os
import numpy as np
from DatasetReader import DatasetReader
from FreemanEncoder import FreemanEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from pyxdameraulevenshtein import damerau_levenshtein_distance as edit_dist


class KNN(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.dsr = DatasetReader()
        self.fenc = FreemanEncoder()
        self.training_data = []

    def generate_labelled_sequences(self, freeman_codes_dict):
        labelled_sequences = []
        codes_list = freeman_codes_dict.items()
        for tup in codes_list:
            for code in tup[1]:
                labelled_sequences.append((tup[0],code))

        return labelled_sequences

    def prepare_data(self, datas, training=[], test=[], split=0.80):
        # Separate data into 2 sets, 1 is training and 1 is test,split is the ratio (the default is 0.70)
        for data in range(len(datas)-1):
            if random.random() < split:
                training.append(datas[data])
            else:
                test.append(datas[data])

    def get_neighbors(self, training, test_instance, k):
        # Get the list of nearest neighbors to a test instance
        distances =[]
        for i in range(len(training)-1):
            dist = edit_dist(test_instance, training[i][1])
            distances.append((training[i], dist))

        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(0, k):
            neighbors.append(distances[x][0])

        return neighbors

    def get_label(self, neighbors):
        # Determine the label of a test instance base on its nearest neighbors
        max = 0
        labels = {}
        for neighbor in neighbors:
            if neighbor[0] not in labels:
                labels[neighbor[0]] = 1
            else:
                labels[neighbor[0]] += 1

        sorted_labels = sorted(labels.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_labels[0][0]

    def evaluation(self, training, test):
        # Evaluate the accuracy of knn
        correct_count = 0
#         k = int(math.ceil(len(training)/10))
        k = 1
        for test_data in test:
            neighbors = self.get_neighbors(training, test_data[1], k)
            label = self.get_label(neighbors)
            if int(label) == int(test_data[0]):
                correct_count += 1

        print (float(correct_count)/len(test))*100

    def knn_train(self, dataset_path, train_test_split=0.8):
        dataset = self.dsr.read_dataset_images(dataset_path)
        freeman_codes_dict = self.fenc.encode_freeman_dataset(dataset)
        labelled_sequences = self.generate_labelled_sequences(freeman_codes_dict)
        training = []
        test = []
        # print labelled_sequences
        self.prepare_data(labelled_sequences,training,test, split=train_test_split)
        self.training_data = training
        
        if train_test_split != 1.0:
            print "Training:" + len(training).__str__()
            print "Test:" + len(test).__str__()
            self.evaluation(training,test)
        
    def knn_predict_one(self, image, k=1):
        if os.path.isfile(image):
            image_array = self.dsr.read_img_bw(image)
            test = self.fenc.encode_freeman(image_array)
        else:
            test = image
        
        # Try to find the nearest neighbors of the first sequences in training
        neighbors = self.get_neighbors(self.training_data, test, k)
        label = self.get_label(neighbors)
        return label

class KNN_strings(object):
    '''
    classdocs
    '''

    def __init__(self, n_neighbors=1):
        '''
        Constructor
        '''
        self.dsr = DatasetReader()
        self.fenc = FreemanEncoder()
        self.data = []
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto', metric=self.lev_metric)
        
    def lev_metric(self, x, y):
        i, j = int(x[0]), int(y[0])     # extract indices
#         if self.data[i] == self.data[j]:
#             print self.data[i], self.data[j], edit_dist(self.data[i], self.data[j])
        return edit_dist(self.data[i], self.data[j])
    
    def knn_train(self, dataset, cv=1, datasplit=0.7):
        
        images_dataset= self.dsr.read_dataset_images(dataset)
        freeman_code_dict = self.fenc.encode_freeman_dataset(images_dataset)
        _, codes, labels = self.dsr.gen_labelled_arrays(freeman_code_dict)
        
        self.data = codes
        
        X = np.arange(len(self.data)).reshape(-1, 1)
        
        if cv <= 1:
            self.knn.fit(X, labels)
        elif cv > 1:
            cv_result = cross_validation.cross_val_score(self.knn, X, labels, cv=cv)
            print cv_result
            
        print 'Training Done!'
            
    def knn_predict(self, test_data, score=False):
        images_dataset= self.dsr.read_dataset_images(test_data)
        freeman_code_dict = self.fenc.encode_freeman_dataset(images_dataset)
        _, codes, labels = self.dsr.gen_labelled_arrays(freeman_code_dict)
        
        X_pred = np.arange(len(codes)).reshape(-1, 1)
        predictions = self.knn.predict(X_pred)
            
        if score == True:
            accuracy = self.knn.score(X_pred, labels)
            print "Test Accuracy: ", accuracy
        
        return predictions
    
    def knn_predict_one(self, test_image):
        image_code = self.fenc.encode_freeman(test_image)
        print image_code
        data = [image_code]
        X_pred = np.arange(len(data)).reshape(-1, 1)
        prediction = self.knn.predict(X_pred)
    
        return prediction


# # knn = KNN_strings(n_neighbors=1)
# knn = KNN()
# for x in range(1):
#     knn.knn_train('I:/eclipse_workspace/CharacterRecognition/teams_dataset', 1.0)
#     print knn.knn_predict_one('I:/eclipse_workspace/CharacterRecognition/omar_dataset/4/canvas_1.jpg')
#     print '==================================================================================='