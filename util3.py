from __future__ import division
import numpy as np
from graphics import GraphWin,Image, Point
import matplotlib.pyplot as plt
from imageio import imread as imread
from imageio import imwrite as imsave
from skimage.transform import resize as imresize
import tkinter as tk
from tkinter import simpledialog

def collectLabel(dimclass):
    ROOT = tk.Tk()
    ROOT.withdraw()
    USER_INP = simpledialog.askstring(title="Class Label",
                                      prompt="Please type in class label:")
    
    classlist = []
    for str in USER_INP.split(','):
        classlist.append(int(str))
    if len(classlist)!=dimclass:
        return []
    return classlist

def imshow2(datas, filename, dimclass, h1, w1):
    plt.figure()
    plt.imshow(datas, vmin=0, vmax=dimclass)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.cla()
    
    img = imread(filename)
    img = imresize(img, (h1,w1))
    imsave(filename, img)

def clickXY(filename, h1, w1, nclicks):
    win = GraphWin("click me!", w1, h1)
    myImage = Image(Point(int(w1/2),int(h1/2)), filename)
    print("imagesize:",myImage.getWidth(), myImage.getHeight())
    myImage.draw(win)

    clicks = []
    for i in range(nclicks):
        p = win.getMouse()
        clicks.append([int(p.getX()), int(p.getY())])
    win.close()
    
    return clicks
