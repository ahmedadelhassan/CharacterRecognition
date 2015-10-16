'''
Created on Oct 15, 2015
Extract freeman code of an image boundary.
@author: ahmad
'''
import os,sys
from PIL import Image
import numpy
import math
import matplotlib.pyplot as plt
from skimage import segmentation, measure
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from Tkinter import Tk
from tkFileDialog import askopenfilename

#Define global variables
#WATCH_OUT! The +x is right, but +y is down, so the representation is different from normal quadrants 
freeman_dict = {-90:'0', -45:'1', 0:'2', 45:'3', 90:'4', 135:'5', 180:'6', -135:'7'}
allowed_directions = numpy.array([0, 45, 90, 135, 180, -45, -90, -135])

def find_nearest(array,value):
    '''
    Find the nearest element of array to the given value
    '''
    idx = (numpy.abs(array-value)).argmin()
    return array[idx]

def encode_freeman(image_contour):
    '''
    Encode the image contour in an 8-direction freeman chain code based on angles
    '''
    freeman_code = ""
    global freeman_dict, allowed_directions
    
    for i in range(len(image_contour) - 1):
        delta_x = image_contour[i+1][1] - image_contour[i][1] 
        delta_y = image_contour[i+1][0] - image_contour[i][0]
        angle = math.degrees(math.atan2(delta_y,delta_x))
        angle = find_nearest(allowed_directions, angle)
        print angle
        if not(delta_x == 0 and delta_y == 0):
            freeman_code += freeman_dict[angle]
            
    return freeman_code

def main():
    
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    image_path = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    
    #read image and convert to matrix
    image = Image.open(image_path)
    image_array = image.getdata()
    
    image_array = numpy.array(image_array).astype(numpy.uint8).reshape((image.size[0],image.size[1]))
    
    #Threshold the image to binary
    thresh = threshold_otsu(image_array)
    image_array = image > thresh
    image_array = ~image_array
    
    #Extract the longest contour in the image
    contours = measure.find_contours(image_array, 0.9)
    contours_main = max(contours, key=len)
    
    # Display the image and plot the main contour found
    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap=plt.cm.gray)
    ax.plot(contours_main[:, 1], contours_main[:, 0], linewidth=2)
    
    # Extract freeman code from contour
    freeman_code = encode_freeman(contours_main)
    
    print freeman_code, len(freeman_code)
    
    plt.show()

if __name__ == '__main__':
    main()