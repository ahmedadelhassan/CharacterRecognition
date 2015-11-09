'''
Created on Oct 22, 2015

@author: tho
'''
import nltk,editdistance,random, math, operator
import numpy as np
from DatasetReader import DatasetReader
from FreemanEncoder import FreemanEncoder
from sklearn.neighbors import KNeighborsRegressor


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
            dist = editdistance.eval(test_instance, training[i][1])
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
        k = int(math.ceil(len(training)/10))
        for test_data in test:
            neighbors = self.get_neighbors(training, test_data[1], k)
            label = self.get_label(neighbors)
            if int(label) == int(test_data[0]):
                correct_count += 1

        print (float(correct_count)/len(test))*100

    def knn_train(self, dataset_path):
        dataset = self.dsr.read_dataset_images(dataset_path)
        freeman_codes_dict = self.fenc.encode_freeman_dataset(dataset)
        labelled_sequences = self.generate_labelled_sequences(freeman_codes_dict)
        training = []
        test = []
        # print labelled_sequences
        self.prepare_data(labelled_sequences,training,test)
        print "Training:" + len(training).__str__()
        print "Test:" + len(test).__str__()

        # Because we need to find 10 clusters so the number of k is the number of training is divided by 10
        k = int(math.ceil(len(training)/10))
        # print k

        # # This instance's label is 5
        # intance = '66666677667767766707557777711001333333233233233223322211211211111111101100000077117777076777767766666555570770000110110001100013311112222332222222222222222222211222234445666666666666666666666666666677654445544554322233223323335531133234455311343355333344445545775334455555654565565555'
        # # Try to find the nearest neighbors of the first sequences in training
        # neighbors = self.get_neighbors(training, intance, k)
        # label = self.get_label(neighbors)
        # print label
        self.evaluation(training,test)



knn = KNN()
for x in range(30):
    knn.knn_train('/home/thovo/PycharmProjects/CharacterRecognition/digits_dataset')
    print '==================================================================================='

# data = '66700110011011110111111111111121011111111111211111121111112111111112111111211112111121122111766666677667766677666776666776667766677666667766666776666677666667766666677667110122332222223322222332222233222223322222332223322233222332222332223322223322223322333334565565565565566556555555655555555655555565555556555555555555555555555555555555555455577533445545'
# print tknn.predict(['66700011011000110001176555555557777711111112111011133223444445544444455445555445'])
