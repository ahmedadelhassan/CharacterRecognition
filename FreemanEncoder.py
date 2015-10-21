'''
Created on Oct 19, 2015
Extract image boundaries, and encode them into freeman chain codes (8-directions)
@author: ahmad
'''

import numpy
import math
from skimage import measure

class FreemanEncoder(object):
    '''
    classdocs
    '''
    #Class global variables
    #WATCH_OUT! The +x is right, but +y is down, so the representation is different from normal quadrants 
    FREEMAN_DICT = {-90:'0', -45:'1', 0:'2', 45:'3', 90:'4', 135:'5', 180:'6', -135:'7'}
    ALLOWED_DIRECTIONS = numpy.array([0, 45, 90, 135, 180, -45, -90, -135])

    def __init__(self):
        '''
        Constructor
        '''
        
    def find_nearest(self, array, value):
        '''
        Find the nearest element of array to the given value
        '''
        idx = (numpy.abs(array-value)).argmin()
        return array[idx]
    
    def encode_freeman(self, image_array):
        '''
        Encode the image contour in an 8-direction freeman chain code based on angles
        '''
        image_contour = self.get_contours(image_array)
        freeman_code = ""
        
        for i in range(len(image_contour) - 1):
            delta_x = image_contour[i+1][1] - image_contour[i][1] 
            delta_y = image_contour[i+1][0] - image_contour[i][0]
            angle = math.degrees(math.atan2(delta_y,delta_x))
            angle = self.find_nearest(self.ALLOWED_DIRECTIONS, angle)
            
            if not(delta_x == 0 and delta_y == 0):
                freeman_code += self.FREEMAN_DICT[angle]
                
        return freeman_code
    
    def get_contours(self, image_array):
        #Extract the longest contour in the image
        contours = measure.find_contours(image_array, 0.9)
        contours_main = max(contours, key=len)
        
        return contours_main
    
    def encode_freeman_dataset(self, images_dataset):
        '''
        Encode images dataset (given as a dictionary where keys are classes, 
        and values are the arrays of images, into a freeman code dictionary of 
        the same structure
        '''
        freeman_code_dict = dict((key, []) for key in images_dataset.keys())
        for key in images_dataset:
            for image in images_dataset[key]:
                image_freeman = self.encode_freeman(image)
                freeman_code_dict[key].append(image_freeman)
                
        return freeman_code_dict
