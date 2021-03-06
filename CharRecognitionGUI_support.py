#! /usr/bin/env python
#
# Support module generated by PAGE version 4.5
# In conjunction with Tcl version 8.6
#    Nov 21, 2015 03:31:59 PM


import sys

#CONGIGURATIONS
combobox = 'kNN'
combobox2 = '0'
save_dir = './test_dataset/'
thumbnails_path = './thumbnails'
training_dataset = './teams_dataset'

try:
    from Tkinter import *
except ImportError:
    from tkinter import *

try:
    import ttk
    py3 = 0
except ImportError:
    import tkinter.ttk as ttk
    py3 = 1

def set_Tk_var():
    # These are Tk variables used passed to Tkinter and must be
    # defined before the widgets using them are created.
    global combobox, combobox2
    combobox = StringVar()
    combobox2 = StringVar()


def init(top, gui, arg=None):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None


