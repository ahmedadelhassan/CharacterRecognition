import numpy
import matplotlib
try:
    from Tkinter import *
except ImportError:
    from tkinter import *
try:
    import ttk
    py3 = 0
except ImportError:
    import tkinter.ttk
    py3 = 1
from tkFileDialog import *
import PIL
import cv2
import skimage
import sklearn
import nltk
import pyxdameraulevenshtein