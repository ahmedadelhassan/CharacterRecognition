'''
Created on Oct 19, 2015
Extract image boundaries, and encode them into freeman chain codes (8-directions)
@author: ahmad
'''

import numpy
import math
from skimage import measure
from copy import deepcopy, copy
import cv2
import matplotlib.pyplot as plt

class FreemanEncoder(object):
    '''
    classdocs
    '''
    #Class global variables
    #WATCH_OUT! The +x is right, but +y is down, so the representation is different from normal quadrants 
#     FREEMAN_DICT = {-90:'0', -45:'1', 0:'2', 45:'3', 90:'4', 135:'5', 180:'6', -135:'7'}
    FREEMAN_DICT = {-90:'4', -45:'3', 0:'2', 45:'1', 90:'0', 135:'7', 180:'6', -135:'5'}
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
        image_copy = copy(image_array)
        image_contour = self.get_contours(image_copy)
        freeman_code = ""
        
        for i in range(len(image_contour) - 1):
            delta_x = image_contour[i+1][0] - image_contour[i][0] 
            delta_y = image_contour[i+1][1] - image_contour[i][1]
            angle = math.degrees(math.atan2(delta_y,delta_x))
            angle = self.find_nearest(self.ALLOWED_DIRECTIONS, angle)
            
#             if not(delta_x == 0 and delta_y == 0):
#                 freeman_code += self.FREEMAN_DICT[angle]
                
            if delta_x == 0 and delta_y == 0:
                pass
            elif delta_x > 0 and delta_y == 0:
                freeman_code += '2'
            elif delta_x < 0 and delta_y == 0:
                freeman_code += '6'
            elif delta_x == 0 and delta_y > 0:
                freeman_code += '4'
            elif delta_x == 0 and delta_y < 0:
                freeman_code += '0'
            elif delta_x > 0 and delta_y > 0:
                freeman_code += '3'
            elif delta_x > 0 and delta_y < 0:
                freeman_code += '1'
            elif delta_x < 0 and delta_y > 0:
                freeman_code += '5'
            elif delta_x < 0 and delta_y < 0:
                freeman_code += '7'    
                
#             # normalize the code
#             freeman_code = self.normalize_freemancode(freeman_code)
#         print freeman_code
#         print image_contour[0],image_contour[1],image_contour[2] 
#         plt.imshow(image_array)
#         plt.show()
        
        return freeman_code
    
    def get_contours(self, image_array):
        #Extract the longest contour in the image
        _, contours, hierarchy = cv2.findContours(image_array,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
#         contours = measure.find_contours(image_array, 0.9, positive_orientation='high')
        contours_main = max(contours, key=len)
        contours_main = [item for sublist in contours_main for item in sublist] #only for the v2 method
        
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
#         return self.pad_codes(freeman_code_dict)
    
    def count_bagofwords(self, string, bagofwords=set(FREEMAN_DICT.values())):
        bagofwords_count = []
        for char in bagofwords:
            char_count = string.count(char)
            bagofwords_count.append(char_count)
            
        return bagofwords_count
    
    def gen_bagofwords_dict(self, freeman_code_dict):
        bagofwords_dict = dict((key, []) for key in freeman_code_dict.keys())
        for key in freeman_code_dict:
            for code in freeman_code_dict[key]:
                code_bagofwords_count = self.count_bagofwords(code)
                bagofwords_dict[key].append(code_bagofwords_count)
                
        return bagofwords_dict
    
    def normalize_freemancode(self,code):
        min_code = min(code)
        while code[0] != min_code:
            code = code[1:] + code[:1]
         
#         compressed_code = []
#         for i in range(2, len(code)):
#             if code[i] != code[i-1] or code[i] != code[i-2]:
#                 compressed_code.append(code[i])
#               
#         return "".join(compressed_code)
        return code
    
    def pad_codes(self, codes_dict):
        max_len = max(map(len,codes_dict.values()))
        
        padded_codes_dict = deepcopy(codes_dict)
        for key in codes_dict:
            codes = [x.zfill(max_len) for x in codes_dict[key]]
            padded_codes_dict[key] = codes
            
        return padded_codes_dict
                
        
## TESTING CODE (WILL BE REMOVED) ##     
# from DatasetReader import DatasetReader
# from FreemanEncoder import FreemanEncoder
# dsr = DatasetReader()
# fenc = FreemanEncoder()
# dataset = dsr.read_img_bw('I:\\eclipse_workspace\\CharacterRecognition\\observation\\0\\0_1.png')
# codes = fenc.encode_freeman(dataset)
# print codes
# print fenc.gen_bagofwords_dict(codes)