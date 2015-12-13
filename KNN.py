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
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from pyxdameraulevenshtein import damerau_levenshtein_distance as edit_dist
from sklearn.utils import shuffle


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


class KNN_statistic(object):
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


    def prepare_data(self, arrays_data=[], arrays_labels=[], split=0.2):
        # Separate data into 2 sets, 1 is training and 1 is test,split is the ratio (the default is 0.20)
        ad_train, ad_test, al_train, al_test = train_test_split(arrays_data, arrays_labels, test_size=split, random_state=42)
        return ad_train, ad_test, al_train, al_test


    def get_neighbors(self, data, data_label, test_instance, k):
        # Get the list of nearest neighbors to a test instance
        distances = []
        for i in range(len(data)):
            dist = edit_dist(test_instance, data[i])
            distances.append((data[i], data_label[i], dist))

        distances.sort(key=operator.itemgetter(2))
        neighbors = []
        for x in range(0, k):
            neighbors.append([distances[x][0], distances[x][1]])

        return neighbors

    def get_label(self, neighbors):
        # Determine the label of a test instance base on its nearest neighbors
        labels = {}
        for neighbor in neighbors:
            if neighbor[1] not in labels:
                labels[neighbor[1]] = 1
            else:
                labels[neighbor[1]] += 1

        sorted_labels = sorted(labels.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_labels[0][0]

    def evaluation(self, data, data_for_distance_caculation, data_label, data_for_distance_calculation_label, k=3):
        # Evaluate the accuracy of knn
        correct_count = 0
        for instance in range(0, len(data)-1):
            neighbors = self.get_neighbors(data_for_distance_caculation,data_for_distance_calculation_label, data[instance], k)
            label = self.get_label(neighbors)
            if int(label) == int(data_label[instance]):
                correct_count += 1

        return (float(correct_count)/len(data))

    def knn_train(self, dataset_path, train_test_split=0.2):
        dataset = self.dsr.read_dataset_images(dataset_path)
        freeman_codes_dict = self.fenc.encode_freeman_dataset(dataset)
        _, arrays_data, arrays_label = self.dsr.gen_labelled_arrays(freeman_codes_dict)
        arrays_data, arrays_label = shuffle(arrays_data, arrays_label)
        ad_train, ad_test, al_train, al_test = self.prepare_data(arrays_data, arrays_label, split=train_test_split)

        # Cross validation with 5 folds
        kf = KFold(len(ad_train), 5)
        result = 0
        for train_index, test_index in kf:
            ad_train_kfold, ad_test_kfold = ad_train[train_index], ad_train[test_index]
            al_train_kfold, al_test_kfold = al_train[train_index], al_train[test_index]
            result += self.evaluation(ad_test_kfold, ad_train_kfold, al_test_kfold, al_train_kfold, k=2)
        result_average = result/5

        # Result with the training
        result_training = self.evaluation(ad_train, ad_train, al_train, al_train, k=2)

        # Result with the test
        result_test = self.evaluation(ad_test, ad_train, al_test, al_train, k=2)
        return result_average, result_training, result_test


# knn = KNN_strings(n_neighbors=1)
# knn = KNN_statistic()
# results = []
# for x in range(50):
#     result_average, result_training, result_test = knn.knn_train("/home/thovo/PycharmProjects/CharacterRecognition/digits_dataset", 0.2)
#     text = result_average.__str__() + " , " + result_training.__str__() + " , " + result_test.__str__() + "\n"
#     results.append(text)
# 
# 
# f = open("Results/knn.txt", "w")
# for item in results:
#     f.write(item)
# 
# f.close()