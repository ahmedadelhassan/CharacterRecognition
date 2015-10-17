'''
Created on Oct 17, 2015
Read all dataset images from file system and save them as a dictionary in JSON format.
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
import fnmatch
from Tkinter import Tk
from tkFileDialog import askdirectory
from tkFileDialog import asksaveasfile
from PIL import Image
from skimage.filters import threshold_otsu

def read_img_bw(image_file):
    #read image and convert to matrix
    image = Image.open(image_file)
    image_array = image.getdata()
    
    image_array = numpy.array(image_array).astype(numpy.uint8).reshape((image.size[0],image.size[1]))
    
    #Threshold the image to binary
    thresh = threshold_otsu(image_array)
    image_array = image > thresh
    image_array = ~image_array
    
    return image_array

def read_dataset_images(dataset_path):
    images = {}
    for root, dirnames, filenames in os.walk(dataset_path):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            character_class = root.split("\\")[-1]
            try:
                image = read_img_bw(os.path.join(root, filename))
                images[character_class].append(image)
            except:
                image = read_img_bw(os.path.join(root, filename))
                images[character_class] = [image]
    
    return images

def save_dataset_binary(dataset):
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    file_path = asksaveasfile(defaultextension='.npy')
    numpy.save(file_path, dataset)

def main():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    dataset_path = askdirectory()
    
    dataset_dict = read_dataset_images(dataset_path)
    
    save_dataset_binary(dataset_dict)

if __name__ == '__main__':
    main()