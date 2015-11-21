'''
Created on Oct 19, 2015
Read all dataset images from file system and save them as a dictionary in numpy binary format.
The dataset has to have the following heirarchy:
    root..
        0..
            0_1.jpg
            0_2.jpg
            ...
        1..
        2..
        ...
@author: ahmad
'''

import os
import numpy
from Tkinter import Tk
from tkFileDialog import asksaveasfile
from PIL import Image
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

class DatasetReader(object):
    '''
    Class for reading the character dataset of a defined structure
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
    def read_img_bw(self, image_file, array_type="one_zero"):
        '''
        Read an image from a file and return its matrix
        inputs:
            image_file: full path to the image file including extension
            array_type: String with two values:
                "one_zero": matrix is uint8 with only 1s and 0s
                "bool": matrix is bool with True and False
        '''
        #read image and convert to matrix
        image = Image.open(image_file).convert('L')
#         image = image.resize((100,100), Image.LANCZOS)
        image_array = image.getdata()
               
        image_array = numpy.array(image_array).astype(numpy.uint8).reshape((image.size[0],image.size[1]))
#         print image_array.shape
        
        #Threshold the image according to array_type parameter
        if array_type == "bool":
            thresh = threshold_otsu(image_array)
            image_array = image > thresh
            image_array = ~image_array
        else:
            image_array[image_array < 128] = 1
            image_array[image_array >= 128] = 0
            image_array[image_array == 1] = 255
#         plt.imshow(image_array)
#         plt.show()
        return image_array

    def read_dataset_images(self, dataset_path):
        images = {}
        for root, _, filenames in os.walk(dataset_path):
            for filename in filenames:
                if filename.endswith(('.jpg', '.jpeg', '.gif', '.png')):
                    if 'posix' in os.name:
                        character_class = root.split("/")[-1]
                    else:
                        character_class = root.split("\\")[-1]
                    try:
                        image = self.read_img_bw(os.path.join(root, filename))
                        images[character_class].append(image)
                    except KeyError:
                        image = self.read_img_bw(os.path.join(root, filename))
                        images[character_class] = [image]
        
        return images
    
    def gen_labelled_arrays(self, dataset_dict):
        labelled_arrays = []
        codes_list = dataset_dict.items()
        for tup in codes_list:
            for code in tup[1]:
                labelled_arrays.append((tup[0],code))
                
        arrays = numpy.array([x[1] for x in labelled_arrays])
        labels = numpy.array([y[0] for y in labelled_arrays])
        return labelled_arrays, arrays, labels
    
    def save_dataset_binary(self, dataset):
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        file_path = asksaveasfile(defaultextension='.npy')
        numpy.save(file_path, dataset)