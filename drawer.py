#!/usr/bin/python
'''
Created on Oct 11, 2015
A drawer GUI to obtain user input using mouse click and drag.
It has Clear and Save to JPG options (saved image resolution fixed at 100*100 
   pixels)
@author: ahmad
'''

import sys
from Tkinter import Tk, Canvas, Button
from PIL import Image, ImageDraw

# Define global variables
drawing_area = ""
x,y=None,None
count=0
image_cnt = 0
image_name = "canvas_"
# Initialize new PIL Image and a draw handler
PIL_image = Image.new("1", (300, 300), "white")
draw = ImageDraw.Draw(PIL_image)


def quit(event):
    '''
    Event function to quit the drawer window
    '''
    sys.exit()
   
def clear(event):
    '''
    Event function to clear the drawing canvas (draw white fill)
    '''
    global drawing_area, PIL_image, draw
    drawing_area.delete("all")
    PIL_image = Image.new("1", (300, 300), "white")
    draw = ImageDraw.Draw(PIL_image)

def drag(event):
    '''
    Event function to start drawing on canvas when left mouse drag happens
    '''
    global drawing_area, x,y, count, draw
    newx,newy=event.x,event.y
    if x is None:
        x,y=newx,newy
        return
    count+=1
    sys.stdout.write("\revent count %d"%count)
    drawing_area.create_line((x,y,newx,newy), width=5, smooth=True)
    draw.line((x,y,newx,newy), width=10)
    x,y=newx,newy

def drag_end(event):
    '''
    Event function to stop drawing on canvas when mouse drag stops
    '''
    global x,y
    x,y=None,None
   
def save(event):
    '''
    Event function to save the current canvas image in JPG format
    '''
    global PIL_image, image_name, image_cnt
    image_cnt += 1
    file_name = image_name + str(image_cnt) + ".jpg"
    PIL_image = PIL_image.resize((100,100), Image.ANTIALIAS)
    PIL_image.save(file_name)


def main():
    global drawing_area
    
    root = Tk()
    root.title("Drawer")
    drawing_area=Canvas(root, width=300, height=300, bg="white")
     
    # Binding Events to the canvas
    drawing_area.bind("<B1-Motion>",drag)
    drawing_area.bind("<ButtonRelease-1>",drag_end)
    drawing_area.pack()
    
    #Buttons
    #Quit Button
    b1=Button(root,text="Quit")
    b1.pack()
    b1.bind("<Button-1>",quit)
    
    #Clear Button
    b2=Button(root,text="Clear")
    b2.pack()
    b2.bind("<Button-1>",clear)
    
    #Save Button
    b3=Button(root,text="Save")
    b3.pack()
    b3.bind("<Button-1>",save)
    root.mainloop()

if __name__ == "__main__":
    main()
